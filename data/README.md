# data/

Сюда кладём датасет в формате YOLO:
```
images/train, images/val
labels/train, labels/val   # *.txt: class cx cy w h (норм.)
```
Классы — см. ../train/data.yaml. Реальные фото фур + масштабные модели/синтетика.
Не коммитим картинки в git (см. .gitignore) — большие; храним отдельно/в releases.
