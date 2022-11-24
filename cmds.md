export TPU_ZONE=us-central1-a export TPU_SIZE=v3-8 export TPU_NAME=comp1 export BUCKET_NAME=lfp_europe_west4_a export PROJECT_ID=learning-from-play-303306


gcloud compute tpus tpu-vm create $TPU_NAME --zone=$ZONE --accelerator-type=$TPU_SIZE --version=v2-alpha --project=$PROJECT_ID


