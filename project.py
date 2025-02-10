# !git clone https://github.com/ultralytics/yolov5
# %cd yolov5
# !pip install -U -r requirements.txt

import torch
from pathlib import Path
from IPython.display import Image, display
import shutil
import matplotlib.pyplot as plt
from google.colab import files

# Carregar o modelo YOLOv5 pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' é o modelo mais leve
# Definir as classes de interesse (exemplo: 'person' e 'car')
# As classes do COCO começam em 0, então você precisa verificar os IDs de classe para 'person' e 'car' no COCO
# No COCO, 'person' é a classe 0 e 'car' é a classe 2
classes_of_interest = [0, 2]
model.classes = classes_of_interest 

# Caminho para o diretório onde o dataset está armazenado
# Usaremos o dataset COCO (com imagens e anotações) já disponibilizado pelo YOLOv5

# Ajuste o arquivo de configuração para trabalhar apenas com as duas classes de interesse
# Você pode personalizar o arquivo 'data.yaml' para o dataset com as classes 'person' e 'car'

# Exemplo de configuração (esse arquivo precisa estar no diretório correto com os dados)
data_yaml = """"
train: ../datasets/coco/train2017
val: ../datasets/coco/val2017
nc: 2
names: ['person', 'car']
"""

# Salve esse arquivo YAML
with open('data.yaml', 'w') as f:
    f.write(data_yaml)

# Iniciar o treinamento com YOLOv5
# !python train.py --img 640 --batch 16 --epochs 3 --data data.yaml --weights yolov5s.pt --cache

#  Testar o modelo treinado com imagens de teste
# !python val.py --data data.yaml --weights runs/train/exp/weights/best.pt --img 640

# Criar um botão de upload de arquivos
uploaded = files.upload()

# Listar os arquivos carregados
for filename in uploaded.keys():
    print(f'Arquivo carregado: {filename}')
    
# Realizar inferência na imagem carregada
img_path = list(uploaded.keys())[0]  # Pega o primeiro arquivo carregado
results = model(img_path)

# Exibir o resultado da detecção
results.show()