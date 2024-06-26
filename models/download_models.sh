#!/bin/bash

# Do not forget to append CLAP 

############################################################################################################

echo "====================================================================================================="
echo "Downloading OpenL3 model"

urls=(
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb512-3.pb"\
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb512-3.json"\
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb512-3.pb"\
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb512-3.json"\
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb6144-3.pb"\
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb6144-3.json"\
    # "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb6144-3.pb"\
    # "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb6144-3.json"\
)

for url in ${urls[@]}; do
    echo "$url"
    wget $url -P .
done

exit 0
############################################################################################################

echo "====================================================================================================="
echo "Downloading YAMNet model"

urls=(
    "https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.json"\
)

for url in ${urls[@]}; do
    echo "$url"
    wget $url -P .
done

############################################################################################################

echo "====================================================================================================="
echo "Downloading VGGish model"

urls=(
    "https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.pb"\
    "https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.json"\
)

for url in ${urls[@]}; do
    echo "$url"
    wget $url -P .
done

############################################################################################################

echo
echo "====================================================================================================="
echo "Downloading FSD SID Net Model"

urls=(
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg41-tlpf-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg41-tlpf-1.json"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-aps-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-aps-1.json"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf_aps-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf_aps-1.json"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf-1.json"\
)

for url in ${urls[@]}; do
    echo "$url"
    wget $url -P .
done

############################################################################################################

echo "====================================================================================================="
echo "Downloading ImageBind model"

wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth

############################################################################################################

echo "====================================================================================================="
echo "Downloading AudioCLIP..."

# AudioCLIP trained on AudioSet (text-, image- and audio-head simultaneously)
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz -P ./models/
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt -P ./models/
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Partial-Training.pt -P ./models/
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/ESRNXFBSP.pt -P ./models/ -O AudioCLIP-ESRNXFBSP.pt

############################################################################################################

echo "====================================================================================================="
echo "Downloading Wav2CLIP..."

wget https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt -P ./models/

echo "Done!"

# ############################################################################################################

# echo "====================================================================================================="
# echo "Downloading CAV-MAE..."

# wget "https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1" -P ./models/