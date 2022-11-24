export TPU_ZONE=us-central1-a export TPU_SIZE=v3-8 export TPU_NAME=comp1 export BUCKET_NAME=lfp_europe_west4_a export PROJECT_ID=learning-from-play-303306


gcloud compute tpus tpu-vm create $TPU_NAME --zone=$TPU_ZONE --accelerator-type=$TPU_SIZE --version=v2-alpha --project=$PROJECT_ID


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone europe-west4-a --project learning-from-play-303306 -- -L 8888:localhost:8888


pip3 install --upgrade pip
pip3 install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

