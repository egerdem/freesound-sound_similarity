"""This script loads a model with a pre-trained checkpoint and extracts clip level embeddings 
for all the audio files in the FSD50K evaluation dataset."""

import os
import time
import glob
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
from lib.directories import EMBEDDINGS_DIR

# from EAT
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import fairseq
import torchaudio

TRIM_DUR = 30 # seconds

if __name__=="__main__":

    DEFAULT_MODEL_PATH = "/Users/ege/Projects/OGUZ/fairseq_pretrains/eat-base_epoch30.pt"
    DEFAULT_AUDIO_DIR = "/Users/ege/Projects/OGUZ/freesound-sound_similarity/FSD50K_demo"
    DEFAULT_OUTPUT_DIR = "/Users/ege/Projects/OGUZ/output"

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path',
                        type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to model.pt chekpoint.")
    parser.add_argument('--audio_dir',
                        type=str, default=DEFAULT_AUDIO_DIR,
                        help="Path to an audio file or a directory with audio files.")
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default="",
                        help="Path to output directory. Default: "
                             f"{EMBEDDINGS_DIR}/<dataset_name>/<model_name>")
    args = parser.parse_args()

    # Get the model name from models/model_name.pt
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # Load the corresponding model
    if 'CLAP_weights_2023' == model_name:
        print("Setting up Microsoft CLAP model...")
        from msclap import CLAP
        import torch
        model = CLAP(model_fp=args.model_path ,version = '2023', use_cuda=False)
        model_name = 'CLAP_2023'
        # Define embedding extractor function
        def extract_embeddings(model, audio_path):
            # Process
            audio_embeddings = model.get_audio_embeddings(audio_files=[audio_path]).tolist()
            return audio_embeddings
    elif "clap" in model_name.lower():
        print("Setting up Laion CLAP model...")
        from lib.laion_clap import CLAP_Module
        import torch
        # Decide type of CLAP model
        if model_name in ["clap-630k-audioset-fusion-best", "clap-630k-fusion-best"]:
            model = CLAP_Module(enable_fusion=True)
        elif "clap-music_speech_audioset_epoch_15_esc_89.98" == model_name:
            model= CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
        else:
            raise ValueError(f"Unknown CLAP model name: {model_name}")
        # Load the model
        model.load_ckpt(args.model_path)
        # Define embedding extractor function
        def extract_embeddings(model, audio_path):
            # Process
            embeddings = model.get_audio_embedding_from_filelist(x=[audio_path], 
                                                                    use_tensor=False).tolist()
            return embeddings
    elif 'beats' in model_name.lower():
        print("Setting up BEATs model...")
        from lib.beats.BEATs import BEATs, BEATsConfig
        import torch
        import librosa
        # load the pre-trained checkpoints
        checkpoint = torch.load(args.model_path)
        # Load the model
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        # Define embedding extractor function
        def extract_embeddings(model, audio_path):
            # Load the audio file and downsample to 16kHz
            audio = librosa.load(audio_path, sr=16000)[0]
            print(audio.shape)
            audio = torch.tensor(audio).unsqueeze(0)
            padding_mask = torch.zeros(1, audio.shape[1]).bool()
            with torch.no_grad():
                embeddings = model.extract_features(
                    audio, 
                    padding_mask=padding_mask
                    )[0]
            embeddings = embeddings.squeeze(0).cpu().numpy().tolist()
            return embeddings
    elif "imagebind" in model_name.lower():
        print("Setting up ImageBind model...")
        from lib.imagebind import data
        from lib.imagebind.models import imagebind_model
        from lib.imagebind.models.imagebind_model import ModalityType
        import torch
        # Load the model
        model = imagebind_model.imagebind_huge(args.model_path, pretrained=True)
        model.eval()
        model.to("cpu")
        # Define embedding extractor function
        def extract_embeddings(model, audio_path):
            # Load the audio file
            inputs = {
                ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], 'cpu'),
            }
            # Process
            with torch.no_grad():
                embeddings = model(inputs)
            embeddings = embeddings['audio'].cpu().numpy().tolist()
            return embeddings
    elif "audioclip" in model_name.lower():
        print("Setting up AudioCLIP model...")
        from lib.audio_clip.utils.transforms import ToTensor1D
        import librosa
        import torch
        # Load the model
        if "ESRNXFBSP".lower() not in model_name.lower():
            from lib.audio_clip.model import AudioCLIP
            model = AudioCLIP(pretrained=args.model_path).eval()
                # Define embedding extractor function
            def extract_embeddings(model, audio_path):
                # Load the audio file
                audio = librosa.load(audio_path, sr=44100)[0]
                # Trim the audio
                audio = audio[:TRIM_DUR*44100]
                # Bring to the right format
                audio = audio.astype(np.float32)
                audio_transforms = ToTensor1D()
                audio = torch.stack([audio_transforms(audio.reshape(1,-1))])
                # Process
                ((embeddings, _, _), _), _ = model(audio=audio)
                embeddings = embeddings.tolist()
                return embeddings
        else:
            from lib.audio_clip.model.esresnet import ESResNeXtFBSP
            import torch
            model = ESResNeXtFBSP(n_fft=2048,
                                hop_length=561,
                                win_length=1654,
                                window='blackmanharris',
                                normalized=True,
                                onesided=True,
                                spec_height=-1,
                                spec_width=-1,
                                num_classes=527,
                                apply_attention=True,
                                pretrained=args.model_path).eval()
            def extract_embeddings(model, audio_path):
                # Load the audio file
                audio = librosa.load(audio_path, sr=44100)[0]
                # Trim the audio
                audio = audio[:TRIM_DUR*44100]
                # Bring to the right format
                audio = audio.astype(np.float32)
                audio_transforms = ToTensor1D()
                audio = torch.stack([audio_transforms(audio.reshape(1,-1))])
                # Process
                #embeddings = model(audio)
                x = model._forward_pre_processing(audio)
                x = model._forward_features(x)
                embeddings = model._forward_reduction(x)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                #((embeddings, _, _), _), _ = model(audio=audio)
                embeddings = embeddings.tolist()
                return embeddings
    elif "wav2clip" in model_name.lower():
        print("Setting up wav2clip model...")
        import lib.wav2clip_wrapper as wav2clip
        import librosa
        # Load the model
        model = wav2clip.get_model(args.model_path)
        # Define embedding extractor function
        def extract_embeddings(model, audio_path):
            # Load the audio file
            audio = librosa.load(audio_path, sr=44100)[0]
            # Trim the audio
            audio = audio[:TRIM_DUR*44100]
            # Create the embeddings
            embeddings = wav2clip.embed_audio(audio, model).tolist()
            return embeddings

    elif "eat-base_epoch30" in model_name.lower():

        print("Setting up EAT model...")
        import librosa

        checkpoint_dir = args.model_path
        @dataclass
        class UserDirModule:
            user_dir: str

        norm_mean = -4.268
        norm_std = 4.569

        """ To Do """

        model_dir = "./EAT" # change this
        model_path = UserDirModule(model_dir)
        print("model_path:", model_path)
        print("model_dir:", model_dir)
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
        model = model[0]
        model.eval()
        model.cpu()

        granularity = "utterance"

        def extract_embeddings(model, audio_path):
            # Load the audio file
            audio, sr = librosa.load(audio_path, sr=16000)

            def target_length_by_audio_duration(audio,sr):
                fs = 100 # "EAT utilize 100Hz fbank features"
                audio_length = int(len(audio) / sr) # duration of the input audio file in seconds
                feature_length =  audio_length * fs # required length considering the audio file length.
                print("feature_length", feature_length)
                if feature_length  <= 512:
                        target_length = 512
                else:
                    target_length = 1024 * (feature_length // 1024 + 1)
                return(target_length)

            target_length = target_length_by_audio_duration(audio,sr)

            source = torch.from_numpy(audio).float().to(device='cpu')
            # source = torchaudio.functional.resample(source, orig_freq=sr, new_freq=16000).float().to(device='cpu')
            # no need to use the above line as we have already resampled to 16khz with librosa

            source = source - source.mean()
            source = source.unsqueeze(dim=0)
            source = torchaudio.compliance.kaldi.fbank(source, htk_compat=True, sample_frequency=16000,
                                                       use_energy=False,
                                                       window_type='hanning', num_mel_bins=128, dither=0.0,
                                                       frame_shift=10).unsqueeze(dim=0)

            n_frames = source.shape[1] # number of time bins
            print("n_frames:", n_frames, "target length:", target_length)
            print("source shape=", source.shape)
            diff = target_length - n_frames
            print("diff=", diff)
            if diff > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, diff))
                source = m(source)
                print("diff>0, padded. new source shape:", source.shape)

            elif diff < 0:
                source = source[:, 0:target_length, :]
                print("diff<0, sliced. new source shape:", source.shape)

            source = (source - norm_mean) / (norm_std * 2)

            # comment this line to fine-tune an end-to-end model
            with torch.no_grad():
                try:
                    print("dooooo")
                    source = source.unsqueeze(dim=0)  # btz=1
                    print("neeee")
                    if granularity == 'frame':
                        feats = model.extract_features(source, padding_mask=None, mask=False, remove_extra_tokens=True)
                        feats = feats['x'].squeeze(0).cpu().numpy()

                    elif granularity == 'utterance':
                        print("before shape:", print(source.shape))
                        feats = model.extract_features(source, padding_mask=None, mask=False, remove_extra_tokens=False)
                        print("SUCCESS with shape",print(source.shape))
                        feats = feats['x']
                        feats = feats[:, 0].squeeze(0).cpu().numpy()
                    else:
                        raise ValueError("Unknown granularity: {}".format(args.granularity))
                except:
                    print("Error in extracting features from {}".format(audio_path))
                    Exception("Error in extracting features from {}".format(audio_path))

            # Create the embeddings
            embeddings = feats

            # Check if `embeddings` is a NumPy array
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
                # print("Embeddings is a NumPy array, converting to list to make it serializable to json")

            return embeddings

    else:
        raise ValueError(f"Unknown model name: {model_name}.")

    if os.path.isdir(args.audio_dir):
        # Get the list of audio files
        args.audio_dir = os.path.normpath(args.audio_dir)
        audio_paths = glob.glob(os.path.join(args.audio_dir, "*.wav"))
        assert len(audio_paths)>0, f"No audio files found in {args.audio_dir}"
        print(f"Found {len(audio_paths)} audio files in {args.audio_dir}")
    else:
        audio_paths = [args.audio_dir]
        print(f"Found 1 audio file: {args.audio_dir}")

    # Determine the output directory
    if args.output_dir=="":
        # If the default output directory is used add the args.audio_dir to the path
        output_dir = os.path.join(EMBEDDINGS_DIR, 
                                os.path.basename(args.audio_dir),
                                model_name)
    else:
        output_dir = os.path.join(args.output_dir, model_name)
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting the embeddings to: {output_dir}")

    # Process each audio clip
    start_time = time.time()
    for i,audio_path in enumerate(audio_paths):
        # Create the output path
        print("audio_path:", audio_path)
        fname = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{fname}.json")
        # Check if the output file already exists
        if not os.path.exists(output_path):
            try:
                # Extract the embeddings
                embeddings = extract_embeddings(model, audio_path)
                # print(f"embeddings returned {embeddings}")
                print(f"embeddings returned")


                with open(output_path, 'w') as outfile:
                    json.dump({'audio_path': audio_path, 'embeddings': embeddings}, outfile, indent=4)
            except:
                print(f"Error processing {audio_path}")
                print("\n")

        # Print progress
        if (i+1)%1000==0 or i==0 or i+1==len(audio_paths):
            print(f"[{i+1:>{len(str(len(audio_paths)))}}/{len(audio_paths)}]")

    total_time = time.time()-start_time
    # print(f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    # print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

    #############
    # print("Done!\n")