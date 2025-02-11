# SAM 2: 이미지 및 비디오에서 모든 것을 세그먼트화

**[Meta의 AI, FAIR](https://ai.meta.com/research/)**

[Nikhila Ravi](https://nikhilaravi.com/), [Valentin Gabeur](https://gabeur.github.io/), [Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en), [Ronghang Hu](https://ronghanghu.com/), [Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en), [Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en), [Haitham Khedr](https://hkhedr.com/), [Roman Rädle](https://scholar.google.de/citations?user=Tpt57v0AAAAJ&hl=en), [Chloe Rolland](https://scholar.google.com/citations?hl=fr&user=n-SnMhoAAAAJ), [Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en), [Eric Mintun](https://ericmintun.github.io/), [Junting Pan](https://junting.github.io/), [Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en), [Nicolas Carion](https://www.nicolascarion.com/), [Chao-Yuan Wu](https://chaoyuan.org/), [Ross Girshick](https://www.rossgirshick.info/), [Piotr Dollár](https://pdollar.github.io/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/)

[[`논문`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`프로젝트`](https://ai.meta.com/sam2)] [[`데모`](https://sam2.metademolab.com/)] [[`데이터셋`](https://ai.meta.com/datasets/segment-anything-video)] [[`블로그`](https://ai.meta.com/blog/segment-anything-2)] [[`BibTeX`](#citing-sam-2)]

![SAM 2 아키텍처](assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)**는 이미지와 비디오에서 프롬프트 가능한 시각적 세그먼트화를 해결하기 위한 기초 모델입니다. 우리는 단일 프레임의 비디오로 이미지를 고려하여 SAM을 비디오로 확장합니다. 모델 설계는 실시간 비디오 처리를 위한 스트리밍 메모리를 갖춘 간단한 트랜스포머 아키텍처입니다. 우리는 사용자 상호작용을 통해 모델과 데이터를 개선하는 모델-인-루프 데이터 엔진을 구축하여 [**우리의 SA-V 데이터셋**](https://ai.meta.com/datasets/segment-anything-video)을 수집합니다. 이는 현재까지 가장 큰 비디오 세그먼트화 데이터셋입니다. 우리의 데이터로 훈련된 SAM 2는 다양한 작업과 시각적 도메인에서 강력한 성능을 제공합니다.

![SA-V 데이터셋](assets/sa_v_dataset.jpg?raw=true)

## 최신 업데이트

**2024년 12월 11일 -- 주요 VOS 속도 향상을 위한 전체 모델 컴파일 및 다중 객체 추적을 더 잘 처리하기 위한 새로운 `SAM2VideoPredictor`**

- 이제 비디오에서 전체 SAM 2 모델의 `torch.compile`을 지원하며, `build_sam2_video_predictor`에서 `vos_optimized=True`로 설정하면 VOS 추론 속도가 크게 향상됩니다.
- 우리는 `SAM2VideoPredictor`의 구현을 업데이트하여 독립적인 객체별 추론을 지원하며, 다중 객체 추적을 위한 프롬프트 가정을 완화하고 추적이 시작된 후 새로운 객체를 추가할 수 있습니다.
- 자세한 내용은 [`RELEASE_NOTES.md`](RELEASE_NOTES.md)를 참조하세요.

**2024년 9월 30일 -- SAM 2.1 개발자 스위트 (새로운 체크포인트, 훈련 코드, 웹 데모) 출시**

- 개선된 모델 체크포인트의 새로운 스위트 (SAM 2.1로 명명됨)가 출시되었습니다. 자세한 내용은 [모델 설명](#model-description)을 참조하세요.
  * 새로운 SAM 2.1 체크포인트를 사용하려면 이 저장소의 최신 모델 코드가 필요합니다. 이전 버전의 저장소를 설치한 경우, 먼저 `pip uninstall SAM-2`를 통해 이전 버전을 제거하고, 이 저장소에서 최신 코드를 가져온 후 (`git pull`), 아래 [설치](#installation)를 따라 저장소를 다시 설치하세요.
- 훈련 (및 미세 조정) 코드가 출시되었습니다. 시작하는 방법은 [`training/README.md`](training/README.md)를 참조하세요.
- SAM 2 웹 데모의 프론트엔드 + 백엔드 코드가 출시되었습니다. 자세한 내용은 [`demo/README.md`](demo/README.md)를 참조하세요.

## 설치

SAM 2는 사용 전에 먼저 설치되어야 합니다. 이 코드는 `python>=3.10`, `torch>=2.5.1` 및 `torchvision>=0.20.1`을 필요로 합니다. PyTorch 및 TorchVision 종속성을 설치하려면 [여기](https://pytorch.org/get-started/locally/)의 지침을 따르세요. GPU 머신에 SAM 2를 설치하려면 다음을 사용하세요:

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```
If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

To use the SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required and can be installed by:

```bash
pip install -e ".[notebooks]"
```

Note:
1. It's recommended to create a new Python environment via [Anaconda](https://www.anaconda.com/) for this installation and install PyTorch 2.5.1 (or higher) via `pip` following https://pytorch.org/. If you have a PyTorch version lower than 2.5.1 in your current environment, the installation command above will try to upgrade it to the latest PyTorch version using `pip`.
2. The step above requires compiling a custom CUDA kernel with the `nvcc` compiler. If it isn't already available on your machine, please install the [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.
3. If you see a message like `Failed to build the SAM 2 CUDA extension` during installation, you can ignore it and still use SAM 2 (some post-processing functionality may be limited, but it doesn't affect the results in most cases).

Please see [`INSTALL.md`](./INSTALL.md) for FAQs on potential issues and solutions.

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)

Then SAM 2 can be used in a few lines as follows for image and video prediction.

### Image prediction

SAM 2 has all the capabilities of [SAM](https://github.com/facebookresearch/segment-anything) on static images, and we provide image prediction APIs that closely resemble SAM for image use cases. The `SAM2ImagePredictor` class has an easy interface for image prompting.

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

Please refer to the examples in [image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)) for static image use cases.

SAM 2 also supports automatic mask generation on images just like SAM. Please see [automatic_mask_generator_example.ipynb](./notebooks/automatic_mask_generator_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb)) for automatic mask generation in images.

### Video prediction

For promptable segmentation and tracking in videos, we provide a video predictor with APIs for example to add prompts and propagate masklets throughout a video. SAM 2 supports video inference on multiple objects and uses an inference state to keep track of the interactions in each video.

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

Please refer to the examples in [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb)) for details on how to add click or box prompts, make refinements, and track multiple objects in videos.

## Load from 🤗 Hugging Face

Alternatively, models can also be loaded from [Hugging Face](https://huggingface.co/models?search=facebook/sam2) (requires `pip install huggingface_hub`).

For image prediction:

```python
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

For video prediction:

```python
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

## Model Description

### SAM 2.1 checkpoints

The table below shows the improved SAM 2.1 checkpoints released on September 29, 2024.
|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2.1_hiera_tiny <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt))    |     38.9     |          91.2          |        76.5         |        71.8        |       77.3        |
|   sam2.1_hiera_small <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt))   |      46      |          84.8          |        76.6         |        73.5        |       78.3        |
| sam2.1_hiera_base_plus <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)) |     80.8     |        64.1          |        78.2         |        73.7        |       78.2        |
|   sam2.1_hiera_large <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt))   |    224.4     |          39.5          |        79.5         |        74.6        |       80.6        |

### SAM 2 checkpoints

The previous SAM 2 checkpoints released on July 29, 2024 can be found as follows:

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2_hiera_tiny <br /> ([config](sam2/configs/sam2/sam2_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt))   |     38.9     |          91.5          |        75.0         |        70.9        |       75.3        |
|   sam2_hiera_small <br /> ([config](sam2/configs/sam2/sam2_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt))   |      46      |          85.6          |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus <br /> ([config](sam2/configs/sam2/sam2_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)) |     80.8     |     64.8    |        74.7         |        72.8        |       75.8        |
|   sam2_hiera_large <br /> ([config](sam2/configs/sam2/sam2_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt))   |    224.4     | 39.7 |        76.0         |        74.6        |       79.8        |

Speed measured on an A100 with `torch 2.5.1, cuda 12.4`. See `benchmark.py` for an example on benchmarking (compiling all the model components). Compiling only the image encoder can be more flexible and also provide (a smaller) speed-up (set `compile_image_encoder: True` in the config).
## Segment Anything Video Dataset

See [sav_dataset/README.md](sav_dataset/README.md) for details.

## Training SAM 2

You can train or fine-tune SAM 2 on custom datasets of images, videos, or both. Please check the training [README](training/README.md) on how to get started.

## Web demo for SAM 2

We have released the frontend + backend code for the SAM 2 web demo (a locally deployable version similar to https://sam2.metademolab.com/demo). Please see the web demo [README](demo/README.md) for details.

## License

The SAM 2 model checkpoints, SAM 2 demo code (front-end and back-end), and SAM 2 training code are licensed under [Apache 2.0](./LICENSE), however the [Inter Font](https://github.com/rsms/inter?tab=OFL-1.1-1-ov-file) and [Noto Color Emoji](https://github.com/googlefonts/noto-emoji) used in the SAM 2 demo code are made available under the [SIL Open Font License, version 1.1](https://openfontlicense.org/open-font-license-official-text/).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 2 project was made possible with the help of many contributors (alphabetical):

Karen Bergan, Daniel Bolya, Alex Bosenberg, Kai Brown, Vispi Cassod, Christopher Chedeau, Ida Cheng, Luc Dahlin, Shoubhik Debnath, Rene Martinez Doehner, Grant Gardner, Sahir Gomez, Rishi Godugu, Baishan Guo, Caleb Ho, Andrew Huang, Somya Jain, Bob Kamma, Amanda Kallet, Jake Kinney, Alexander Kirillov, Shiva Koduvayur, Devansh Kukreja, Robert Kuo, Aohan Lin, Parth Malani, Jitendra Malik, Mallika Malhotra, Miguel Martin, Alexander Miller, Sasha Mitts, William Ngan, George Orlin, Joelle Pineau, Kate Saenko, Rodrick Shepard, Azita Shokrpour, David Soofian, Jonathan Torres, Jenny Truong, Sagar Vaze, Meng Wang, Claudette Ward, Pengchuan Zhang.

Third-party code: we use a GPU-based connected component algorithm adapted from [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch) (with its license in [`LICENSE_cctorch`](./LICENSE_cctorch)) as an optional post-processing step for the mask predictions.

## Citing SAM 2

If you use SAM 2 or the SA-V dataset in your research, please use the following BibTeX entry.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
