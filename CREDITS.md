# Credits and Data Attribution

Third-party data and models used in this project, with the attribution their licences require.

## Datasets

**Ghost Pot Side-Scan Sonar Detection Dataset**
PINGEcosystem. `PINGEcosystem/sss-crab-pot-detection-ds`, Hugging Face.
DOI: [10.57967/hf/8397](https://doi.org/10.57967/hf/8397)
Licence: CC-BY-SA-4.0
Side-scan sonar imagery of derelict crab pots collected in Delaware's Inland Bays and Delaware Bay.

**SCTD — Sonar Common Target Detection Dataset**
<https://github.com/MingqiangNing/SCTD>
357 images, 363 annotated objects (ship, aircraft, human), Pascal VOC format.
Cite per the repository's stated requirement if results using it are published.

## Models

**GhostVision detection models**
PINGEcosystem: `gv-yolo12`, `gv-yolo26`, `gv-rf-detr`, Hugging Face.
Licence: CC-BY-SA-4.0
Fine-tuned detectors for derelict gear in side-scan sonar, trained on the dataset above.

Publication:
> Bodine, C. S., Baxevani, K., Abbasi, N., Wierzbicki, J., Christoph, O., Hughes, C.,
> Bagoren, O., Hines, O., Greco, J., & Trembanis, A. (2026).
> *GhostVision: Democratizing Derelict Gear Detection Using Low-Cost Sonar and
> Artificial Intelligence.* Journal of Marine Science and Engineering, 14(10), 951.

Project: <https://github.com/PINGEcosystem/GhostVision>

**YOLOv8 pretrained weights**
Ultralytics, AGPL-3.0. Used as the COCO-pretrained initialisation for fine-tuning.

## Models distributed with this project

`backend/weights/yolo/debris_crabpot.pt` was trained on the crab-pot dataset above and is
therefore a **derivative work under CC-BY-SA-4.0**. Redistributing it, or publishing results
from it, carries the attribution and share-alike terms of that licence. Cite the dataset DOI
and the GhostVision publication.

`backend/weights/yolo/best.pt` and `backend/weights/unet/unet.pth` were trained on SCTD, which
asks for citation rather than share-alike.

## Share-alike notice

CC-BY-SA-4.0 applies share-alike terms to derivative works. Any model we train on the
crab-pot dataset, and any redistribution of these models, inherits that obligation.
Check this before publishing or distributing trained weights.
