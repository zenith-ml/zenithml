import abc
import os
from itertools import chain
from pathlib import Path
from typing import List, Optional, Dict, Union

from zenithml.gcp import BQRunner


def _generate_parquet_export(all_analyzers, bq_table, where_clause, output_path, renew_cache):
    analyze_data = {}
    for pp in all_analyzers:
        if pp.export_as_parquet:
            data = pp.export_to_parquet(bq_table, where_clause, output_path, renew_cache)
            output_path.mkdir(exist_ok=True, parents=True)
            analyze_data.update(data)
    return analyze_data


def _generate_query_str(analyzers, bq_table, where_clause):
    sub_queries = [pp.analyze_subquery(bq_table, where_clause) for pp in analyzers]
    sub_queries_txt = ",".join([q for q in sub_queries if q is not None])
    qtxt = f"WITH {sub_queries_txt} \n" if sub_queries_txt else ""
    fields = ",\n\t".join(list(chain(*[pp.analyze_query_fields() for pp in analyzers])))
    if where_clause:
        qtxt += f"""SELECT COUNT(*) AS __count, \n\t {fields} \n\t FROM `{bq_table}` WHERE {where_clause}"""
    else:
        qtxt += f"""SELECT COUNT(*) AS __count, \n\t {fields} \n\t FROM `{bq_table}`"""
    return qtxt


class BQAnalyzer(abc.ABC):
    def __init__(self, input_col: Union[str, List[str]], feature: str, export_as_parquet: bool = False):
        self.feature = feature
        self.input_col = input_col
        self.export_as_parquet = export_as_parquet
        self.bq_runner = BQRunner()

    def analyze_subquery(self, base_table, where_clause) -> Optional[str]:
        return None

    @abc.abstractmethod
    def analyze_query_fields(self, **kwargs) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def fit(
        layer_list,
        bq_table: str,
        where_clause: Optional[str] = None,
        renew_cache: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ):
        assert output_path is not None, "output_path must be set (check bq_analyzer_out_path) in fit()"
        output_path = output_path if isinstance(output_path, Path) else Path(output_path)
        bq_runner = BQRunner()

        # Construct a query using the analyzer and get the analyze data from BQ
        all_analyzers = [pp.bq_analyzer() for pp in layer_list if pp.bq_analyzer() is not None]
        qtxt = _generate_query_str(all_analyzers, bq_table, where_clause)

        df = bq_runner.query(qtxt, renew_cache=renew_cache)
        schema = bq_runner.bq_client.get_table(bq_table)
        analyze_data = {k: v[0] for k, v in df.to_dict().items()}

        analyze_data.update(_generate_parquet_export(all_analyzers, bq_table, where_clause, output_path, renew_cache))

        for layer in layer_list:
            layer.load(analyze_data)
        return analyze_data, schema

    def export_to_parquet(self, base_table, where_clause, output_path, renew_cache: bool = False) -> Dict[str, str]:
        return {}


class StandardScalerBQAnalyzer(BQAnalyzer):
    def analyze_query_fields(self, **kwargs) -> List[str]:
        """Calculate percentile thresholds for log preprocess"""
        return [
            f"AVG({self.input_col}) AS {self.feature}_avg",
            f"MIN({self.input_col}) AS {self.feature}_min",
            f"MAX({self.input_col}) AS {self.feature}_max",
            f"STDDEV({self.input_col}) AS {self.feature}_stddev",
        ]


class LogScalerBQAnalyzer(BQAnalyzer):
    def __init__(self, input_col: Union[str, List[str]], feature: str, percentile: float, **kwargs):
        super().__init__(input_col=input_col, feature=feature, **kwargs)
        self.percentile = percentile

    def analyze_subquery(self, base_table, where_clause) -> Optional[str]:
        where_clause = (
            f"{self.input_col} > 0" if where_clause is None else f"({where_clause})" + f" AND {self.input_col}  > 0"
        )
        return f"""
            {self.feature}_cat_subq AS (
                SELECT APPROX_QUANTILES({self.input_col}, 100) as fnum
            FROM `{base_table}`
            WHERE {where_clause})
        """

    def analyze_query_fields(self, **kwargs) -> List[str]:
        """Calculate percentile thresholds for log preprocess"""
        return [
            f"ARRAY(SELECT fnum[OFFSET({self.percentile})] FROM {self.feature}_cat_subq)[OFFSET(0)] AS {self.feature}_percentile",
        ]


class BucketizedBQAnalyzer(BQAnalyzer):
    def __init__(self, input_col: Union[str, List[str]], feature: str, bins: int, **kwargs):
        super().__init__(input_col=input_col, feature=feature, **kwargs)
        self.bins = bins

    def analyze_query_fields(self, **kwargs) -> List[str]:
        if self.bins:
            return [
                f"APPROX_QUANTILES({self.input_col}, {self.bins}) AS {self.feature}_bins",
            ]
        else:
            return []


class CategoricalBQAnalyzer(BQAnalyzer):
    def __init__(self, input_col: Union[str, List[str]], feature: str, top_k: int, export_as_parquet: bool = False):
        super().__init__(input_col=input_col, feature=feature, export_as_parquet=export_as_parquet)
        self.top_k = top_k

    def analyze_subquery(self, base_table, where_clause) -> Optional[str]:
        where_clause = (
            f"{self.input_col} IS NOT NULL"
            if where_clause is None
            else f"({where_clause})" + f" AND {self.feature} IS NOT NULL"
        )
        return f"""
            {self.feature}_cat_subq AS (
                SELECT {self.input_col} AS {self.feature}, COUNT(*) as freq
            FROM `{base_table}`
            WHERE {where_clause}
            GROUP BY {self.input_col}
            ORDER BY freq DESC LIMIT {self.top_k})
        """

    def analyze_query_fields(self, **kwargs) -> List[str]:
        if not self.export_as_parquet:
            return [
                f"""ARRAY(SELECT {self.feature} FROM
                    {self.feature}_cat_subq) AS {self.feature}_cat"""
            ]
        else:
            return []

    def export_to_parquet(self, base_table, where_clause, output_path, renew_cache: bool = False):
        query = f"""WITH {self.analyze_subquery(base_table, where_clause)}
        SELECT {self.feature} AS {self.feature}_cat  FROM {self.feature}_cat_subq"""

        vocab_path = os.path.join(output_path, f"{self.feature}_cat.vocab")
        if not os.path.exists(vocab_path) or renew_cache:
            self.bq_runner.query(query, renew_cache=renew_cache).to_parquet(vocab_path)

        return {f"{self.feature}_cat": vocab_path}


class WeightedCategoricalBQAnalyzer(BQAnalyzer):
    def __init__(self, input_col: Union[str, List[str]], feature: str, top_k: int, export_as_parquet: bool = False):
        super().__init__(input_col=input_col, feature=feature, export_as_parquet=export_as_parquet)
        self.top_k = top_k

    def analyze_subquery(self, base_table, where_clause) -> Optional[str]:
        where_clause = (
            f"{self.input_col} IS NOT NULL"
            if where_clause is None
            else f"({where_clause})" + f" AND {self.feature} IS NOT NULL"
        )
        return f"""
            {self.feature}_cat_subq AS (
                SELECT {self.input_col} AS {self.feature}, COUNT(*) as freq
            FROM `{base_table}`, UNNEST({self.input_col}) AS {self.input_col}
            WHERE {where_clause}
            GROUP BY {self.input_col}
            ORDER BY freq DESC LIMIT {self.top_k})
        """

    def analyze_query_fields(self, **kwargs) -> List[str]:
        if not self.export_as_parquet:
            return [f"""ARRAY(SELECT {self.feature} FROM {self.feature}_cat_subq) AS {self.feature}_cat"""]

        else:
            return []

    def export_to_parquet(self, base_table, where_clause, output_path, renew_cache: bool = False):
        query = f"""WITH {self.analyze_subquery(base_table, where_clause)}
        SELECT {self.feature} AS {self.feature}_cat_subq  FROM {self.feature}_cat_subq"""

        vocab_path = os.path.join(output_path, f"{self.feature}_cat.vocab")
        if not os.path.exists(vocab_path) or renew_cache:
            self.bq_runner.query(query, renew_cache=renew_cache).to_parquet(vocab_path)

        return {f"{self.feature}_cat": vocab_path}
