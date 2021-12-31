variable "project_id" {
  type    = string
  default = "condortools"
}

variable "framework" {
  type    = string
  default = "torch"
}

variable "version" {
  type    = string
}

variable "image_name" {
  type    = string
  default = "condorml"
}

variable "image_zone" {
  type    = string
  default = "europe-west4-b"
}

source "googlecompute" "gcp-instance" {
  project_id = var.project_id
  source_image_project_id = ["deeplearning-platform-release"]
  source_image_family = "common-cu110-debian-10"
  ssh_username = "packer"
  zone = var.image_zone
  machine_type = "n1-standard-8"
  disk_size = 50
  image_name = join("-", [var.image_name, var.framework, replace(var.version, ".", "-")])
  image_family = var.image_name
  on_host_maintenance = "TERMINATE"
  accelerator_type = "projects/${var.project_id}/zones/${var.image_zone}/acceleratorTypes/nvidia-tesla-v100"
  accelerator_count = 1
  scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  use_internal_ip = false
}

build {
  sources = ["sources.googlecompute.gcp-instance"]

  provisioner "file"{
    source = "launch.sh"
    destination = "/tmp/launch.sh"
  }

  provisioner "shell" {
    inline = [
      "sudo mkdir -p /scripts",
      "sudo chmod -R 755 /scripts",
      "sudo mv /tmp/launch.sh /scripts/launch.sh",
    ]
  }

  provisioner "shell" {
    pause_before = "180s"
    timeout      = "180s"
    inline = [
      "sudo apt-get update && sudo /opt/deeplearning/install-driver.sh",
    ]
  }

  provisioner "shell" {
    timeout      = "1200s"
    inline = [
      "sudo gcloud auth configure-docker",
      join("", ["sudo docker pull condortools/condor-",var.framework, ":", var.version]),
    ]
  }

}