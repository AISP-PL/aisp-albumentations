# aisp-albumentations
Console tool to albumentate images for deep learning. Using great albumentations (https://albumentations.ai/) package.

# Install

```python
pip install -r requirements.txt
./install.sh
```

# Usage

Augment images by color transformations
```shell
python ./main.py -ac -i tests/TestImages1/
```

Augment images by shape transformations
```shell
python ./main.py -as -i tests/TestImages1/
```
