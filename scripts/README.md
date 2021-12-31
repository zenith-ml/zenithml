# Infrastructure Setup

1. **BASE VM**: Create Base VM on GCP or AWS.
2. **RAY CLUSTER**: Copy and Edit ray.*-template.YAML files.
3. **Run**: `ray up PATH_TO_YAML`

### BASE VM

```bash
VER=$(python setup.py -V); packer build \
 --var version=$VER \
 --var project_id=PROJECT_ID \
 --var image_family=IMAGE_FAMILY \
 --var image_name=IMAGE_NAME \
 scripts/vm/gcp.pkr.hcl
```

### RAY CLUSTER
