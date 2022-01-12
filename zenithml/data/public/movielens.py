import nvtabular as nvt
from nvtabular.dispatch import get_lib
from nvtabular.utils import download_file

from zenithml import preprocess as pp
from zenithml.data import ParquetDataset
from zenithml.data.base import BaseDatasetMixin
from zenithml.utils import rich_logging


class Movielens(BaseDatasetMixin):
    def __init__(self, working_dir: str, data_dir: str, variant: str = "1m", **kwargs):
        assert variant.lower() in {"1m", "10m", "25m"}
        super().__init__(
            name=f"{self.__class__.__name__}_{variant.lower()}",
            base_data_dir=data_dir,
            working_dir=working_dir,
        )
        self.variant = variant.lower()
        self.download(verbose=False, **kwargs)

    def info(self):
        return """
        The MovieLens 100K/1M/10M/25M is a popular dataset for recommender systems and is used in academic publications.
        The dataset contains 25M movie ratings for 62,000 movies given by 162,000 users.
        This dataset will only use the user/item/rating information of MovieLens,
        but the original dataset provides additional metadata for the movies, as well.
        """

    def download(self, verbose: bool = False, **kwargs):
        if not (self.data_dir / f"ml-{self.variant}.zip").exists():
            download_file(
                url=f"http://files.grouplens.org/datasets/movielens/ml-{self.variant}.zip",
                local_filename=str(self.data_dir / f"ml-{self.variant}.zip"),
                redownload=False,
            )
            self.init_dataset(**kwargs)
        else:
            if verbose:
                rich_logging().warn(f"Dataset already exists at {str(self.data_dir)}")

    def init_dataset(self, test_split_percentage: float = 0.2):
        df_lib = get_lib()

        if self.variant == "1m":
            movies = df_lib.read_csv(
                self.data_dir / f"ml-{self.variant}/movies.dat",
                sep="::",
                encoding="ISO-8859-1",
                names=["MovieID", "Title", "Genres"],
            )
            users = df_lib.read_csv(
                self.data_dir / f"ml-{self.variant}/users.dat",
                sep="::",
                encoding="ISO-8859-1",
                names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
            )
            ratings = df_lib.read_csv(
                self.data_dir / f"ml-{self.variant}/ratings.dat",
                sep="::",
                encoding="ISO-8859-1",
                names=["UserID", "MovieID", "Rating", "Timestamp"],
            )
            ratings = ratings.rename(
                columns={"UserID": "userId", "MovieID": "movieId", "Rating": "rating", "Timestamp": "timestamp"}
            )
            movies = movies.rename(columns={"MovieID": "movieId", "Title": "title", "Genres": "genres"})
            users = users.rename(
                columns={
                    "UserID": "userId",
                    "Gender": "gender",
                    "Age": "age",
                    "Occupation": "occupation",
                    "Zip-code": "zipcode",
                }
            )
        else:
            movies = df_lib.read_csv(self.data_dir / f"ml-{self.variant}/movies.csv")
            ratings = df_lib.read_csv(self.data_dir / f"ml-{self.variant}/ratings.csv")

        # shuffle the dataset
        ratings = ratings.sample(len(ratings), replace=False)

        # split the train_df as training and validation data sets.
        num_test = int(len(ratings) * test_split_percentage)
        train_dataset, test_dataset = ratings[:-num_test], ratings[-num_test:]

        movies["genres"] = movies["genres"].str.split("|")
        moviesJoined = ["userId", "movieId"] >> nvt.ops.JoinExternal(movies, on=["movieId"])
        userJoined = ["userId", "movieId"] >> nvt.ops.JoinExternal(users, on=["userId"])
        ratings = nvt.ColumnSelector(["rating"]) >> nvt.ops.LambdaOp(lambda col: (col > 3).astype("int8"))
        output = moviesJoined + userJoined + ratings
        workflow = nvt.Workflow(output)

        workflow.transform(nvt.Dataset(train_dataset)).to_parquet(
            output_path=str(self.data_dir / "train"),
            shuffle=nvt.io.Shuffle.PER_PARTITION,
            cats=["userId", "movieId", "genres"],
            labels=["rating"],
        )

        workflow.transform(nvt.Dataset(test_dataset)).to_parquet(
            output_path=str(self.data_dir / "test"),
            shuffle=nvt.io.Shuffle.PER_PARTITION,
            cats=["userId", "movieId", "genres"],
            labels=["rating"],
        )

    def get_preprocessor(self):
        ftransforms = pp.Preprocessor()
        ftransforms.add_outcome_variable("rating")
        ftransforms.add_variable_group(
            "features",
            [
                pp.Categorical("userId"),
                pp.Categorical("movieId"),
                pp.CategoricalList("genres"),
            ],
        )

        return ftransforms

    @property
    def train(self):
        return ParquetDataset(
            path=str(self.data_dir / "train"),
            working_dir=self.working_dir,
            preprocessor=self.get_preprocessor(),
        )

    @property
    def test(self):
        return ParquetDataset(
            path=str(self.data_dir / "test"),
            working_dir=self.working_dir,
            preprocessor=self.get_preprocessor(),
        )

    @property
    def validation(self):
        raise NotImplementedError
