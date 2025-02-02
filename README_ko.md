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
Windows에 설치하는 경우, [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)과 Ubuntu를 사용하는 것이 강력히 권장됩니다.

SAM 2 예측기를 사용하고 예제 노트북을 실행하려면 `jupyter`와 `matplotlib`가 필요하며 다음을 통해 설치할 수 있습니다:

```bash
pip install -e ".[notebooks]"
```

참고:
1. 이 설치를 위해 [Anaconda](https://www.anaconda.com/)를 통해 새로운 Python 환경을 생성하고 https://pytorch.org/를 따라 `pip`을 통해 PyTorch 2.5.1 (또는 그 이상)을 설치하는 것이 권장됩니다. 현재 환경에 PyTorch 버전이 2.5.1보다 낮은 경우, 위의 설치 명령은 `pip`을 사용하여 최신 PyTorch 버전으로 업그레이드하려고 시도할 것입니다.
2. 위 단계는 `nvcc` 컴파일러로 사용자 정의 CUDA 커널을 컴파일하는 것을 요구합니다. 머신에 이미 설치되어 있지 않은 경우, PyTorch CUDA 버전에 맞는 [CUDA 도구킷](https://developer.nvidia.com/cuda-toolkit-archive)을 설치하세요.
3. 설치 중 `Failed to build the SAM 2 CUDA extension`과 같은 메시지가 표시되면 무시하고 SAM 2를 계속 사용할 수 있습니다 (일부 후처리 기능이 제한될 수 있지만 대부분의 경우 결과에 영향을 미치지 않습니다).

잠재적인 문제와 해결책에 대한 FAQ는 [`INSTALL.md`](./INSTALL.md)를 참조하세요.

## 시작하기

### 체크포인트 다운로드

먼저 모델 체크포인트를 다운로드해야 합니다. 모든 모델 체크포인트는 다음을 실행하여 다운로드할 수 있습니다:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

또는 개별적으로 다음에서 다운로드할 수 있습니다:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

(이들은 SAM 2.1로 명명된 개선된 체크포인트입니다; 자세한 내용은 [모델 설명](#model-description)을 참조하세요.)

그런 다음 SAM 2는 이미지 및 비디오 예측을 위해 다음과 같이 몇 줄로 사용할 수 있습니다.

### 이미지 예측

SAM 2는 정적 이미지에서 [SAM](https://github.com/facebookresearch/segment-anything)의 모든 기능을 가지고 있으며, 이미지 사용 사례를 위해 SAM과 유사한 이미지 예측 API를 제공합니다. `SAM2ImagePredictor` 클래스는 이미지 프롬프트를 위한 간단한 인터페이스를 가지고 있습니다.

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

정적 이미지 사용 사례에 대한 예제는 [image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb) (Colab에서도 [여기](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)에서 확인 가능)를 참조하세요.

SAM 2는 또한 SAM과 마찬가지로 이미지에서 자동 마스크 생성을 지원합니다. 이미지에서 자동 마스크 생성을 위한 예제는 [automatic_mask_generator_example.ipynb](./notebooks/automatic_mask_generator_example.ipynb) (Colab에서도 [여기](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb)에서 확인 가능)를 참조하세요.

### 비디오 예측

비디오에서 프롬프트 가능한 세그먼트화 및 추적을 위해, 우리는 비디오 예측기를 제공하며, 예를 들어 프롬프트를 추가하고 비디오 전체에 마스크를 전파하는 API를 제공합니다. SAM 2는 여러 객체에 대한 비디오 추론을 지원하며, 각 비디오의 상호작용을 추적하기 위해 추론 상태를 사용합니다.

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # 새로운 프롬프트를 추가하고 동일한 프레임에서 즉시 출력을 얻습니다
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # 프롬프트를 전파하여 비디오 전체에 마스크를 생성합니다
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

비디오에서 클릭 또는 박스 프롬프트를 추가하고, 수정하고, 여러 객체를 추적하는 방법에 대한 자세한 내용은 [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb) (Colab에서도 [여기](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb)에서 확인 가능)를 참조하세요.

## 🤗 Hugging Face에서 로드하기

대안으로, 모델은 [Hugging Face](https://huggingface.co/models?search=facebook/sam2)에서 로드할 수도 있습니다 (`pip install huggingface_hub` 필요).

이미지 예측의 경우:

```python
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

비디오 예측의 경우:

```python
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # 새로운 프롬프트를 추가하고 동일한 프레임에서 즉시 출력을 얻습니다
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # 프롬프트를 전파하여 비디오 전체에 마스크를 생성합니다
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

## 모델 설명

### SAM 2.1 체크포인트

아래 표는 2024년 9월 29일에 출시된 개선된 SAM 2.1 체크포인트를 보여줍니다.
|      **모델**       | **크기 (M)** |    **속도 (FPS)**     | **SA-V 테스트 (J&F)** | **MOSE 검증 (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2.1_hiera_tiny <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt))    |     38.9     |          91.2          |        76.5         |        71.8        |       77.3        |
|   sam2.1_hiera_small <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt))   |      46      |          84.8          |        76.6         |        73.5        |       78.3        |
| sam2.1_hiera_base_plus <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)) |     80.8     |        64.1          |        78.2         |        73.7        |       78.2        |
|   sam2.1_hiera_large <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt))   |    224.4     |          39.5          |        79.5         |        74.6        |       80.6        |

### SAM 2 체크포인트

2024년 7월 29일에 출시된 이전 SAM 2 체크포인트는 다음과 같습니다:

|      **모델**       | **크기 (M)** |    **속도 (FPS)**     | **SA-V 테스트 (J&F)** | **MOSE 검증 (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2_hiera_tiny <br /> ([config](sam2/configs/sam2/sam2_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt))   |     38.9     |          91.5          |        75.0         |        70.9        |       75.3        |
|   sam2_hiera_small <br /> ([config](sam2/configs/sam2/sam2_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt))   |      46      |          85.6          |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus <br /> ([config](sam2/configs/sam2/sam2_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)) |     80.8     |     64.8    |        74.7         |        72.8        |       75.8        |
|   sam2_hiera_large <br /> ([config](sam2/configs/sam2/sam2_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt))   |    224.4     | 39.7 |        76.0         |        74.6        |       79.8        |

속도는 `torch 2.5.1, cuda 12.4`를 사용한 A100에서 측정되었습니다. 벤치마킹 예제는 `benchmark.py`를 참조하세요 (모델 구성 요소를 모두 컴파일). 이미지 인코더만 컴파일하면 더 유연하고 (작은) 속도 향상을 제공할 수 있습니다 (구성에서 `compile_image_encoder: True`로 설정).

## Segment Anything Video 데이터셋

자세한 내용은 [sav_dataset/README.md](sav_dataset/README.md)를 참조하세요.

## SAM 2 훈련

이미지, 비디오 또는 둘 다의 사용자 정의 데이터셋에서 SAM 2를 훈련하거나 미세 조정할 수 있습니다. 시작하는 방법은 훈련 [README](training/README.md)를 참조하세요.

## SAM 2 웹 데모

SAM 2 웹 데모의 프론트엔드 + 백엔드 코드가 출시되었습니다 (https://sam2.metademolab.com/demo와 유사한 로컬 배포 가능한 버전). 자세한 내용은 웹 데모 [README](demo/README.md)를 참조하세요.

## 라이선스

SAM 2 모델 체크포인트, SAM 2 데모 코드 (프론트엔드 및 백엔드), SAM 2 훈련 코드는 [Apache 2.0](./LICENSE) 라이선스 하에 제공되지만, SAM 2 데모 코드에서 사용된 [Inter Font](https://github.com/rsms/inter?tab=OFL-1.1-1-ov-file)와 [Noto Color Emoji](https://github.com/googlefonts/noto-emoji)는 [SIL Open Font License, version 1.1](https://openfontlicense.org/open-font-license-official-text/) 하에 제공됩니다.

## 기여하기

[기여하기](CONTRIBUTING.md) 및 [행동 강령](CODE_OF_CONDUCT.md)을 참조하세요.

## 기여자

SAM 2 프로젝트는 많은 기여자들의 도움으로 가능했습니다 (알파벳 순):

Karen Bergan, Daniel Bolya, Alex Bosenberg, Kai Brown, Vispi Cassod, Christopher Chedeau, Ida Cheng, Luc Dahlin, Shoubhik Debnath, Rene Martinez Doehner, Grant Gardner, Sahir Gomez, Rishi Godugu, Baishan Guo, Caleb Ho, Andrew Huang, Somya Jain, Bob Kamma, Amanda Kallet, Jake Kinney, Alexander Kirillov, Shiva Koduvayur, Devansh Kukreja, Robert Kuo, Aohan Lin, Parth Malani, Jitendra Malik, Mallika Malhotra, Miguel Martin, Alexander Miller, Sasha Mitts, William Ngan, George Orlin, Joelle Pineau, Kate Saenko, Rodrick Shepard, Azita Shokrpour, David Soofian, Jonathan Torres, Jenny Truong, Sagar Vaze, Meng Wang, Claudette Ward, Pengchuan Zhang.

서드파티 코드: 우리는 마스크 예측을 위한 선택적 후처리 단계로 [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch)에서 적응된 GPU 기반 연결 구성 요소 알고리즘을 사용합니다 (그 라이선스는 [`LICENSE_cctorch`](./LICENSE_cctorch)에 있습니다).

## SAM 2 인용하기

연구에서 SAM 2 또는 SA-V 데이터셋을 사용하는 경우, 다음 BibTeX 항목을 사용하세요.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
