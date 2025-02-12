# Transfer Learning com YOLOv5 para Detecção de Objetos

Este projeto utiliza a técnica de **Transfer Learning** com a rede **YOLOv5** para a **detecção de objetos** em imagens, treinando o modelo para detectar duas classes do **Dataset COCO** sendo elas "person" e "car".

## Objetivo

O objetivo deste projeto é treinar um modelo de detecção de objetos utilizando o **YOLOv5**, uma das arquiteturas mais populares para detecção em tempo real, com um dataset específico de duas classes extraídas do **COCO Dataset**. Através do Transfer Learning, o modelo é inicialmente treinado em um conjunto de dados genérico e depois ajustado para detectar as classes desejadas, melhorando sua performance e acelerando o treinamento.

## Requisitos

Para rodar este projeto, você precisará de:

- Python 3.x
- Google Colab (recomendado) ou ambiente local com suporte a CUDA (para aceleração de GPU)
- Pacotes Python:
  - `torch`
  - `opencv`
  - `matplotlib`
  - `pandas`
  - `yolov5` (Clone do repositório YOLOv5)
  - Outros pacotes necessários (instalados automaticamente com `requirements.txt`)

## Como Rodar o Projeto

### 1. Preparar o Ambiente

Primeiramente, clone o repositório YOLOv5 e instale as dependências:

```bash
!git clone https://github.com/ultralytics/yolov5.git
!pip install -U -r yolov5/requirements.txt
```
### 2. Baixar e Preparar o Dataset COCO
Se você não tiver o dataset COCO, faça o download ou utilize a versão já existente no repositório. Se necessário, ajuste o arquivo de configuração do YOLO para usar apenas as classes desejadas do COCO.

### 3. Configurar o Treinamento
Configure o arquivo coco.yaml (ou crie um próprio) com as duas classes que deseja treinar. Um exemplo básico de configuração de arquivo coco.yaml seria:

```yaml
train: ./path/to/train/images
val: ./path/to/val/images

nc: 2  # Número de classes
names: ['person', 'car']  # Nome das classes
```
### 4. Treinando o Modelo com YOLOv5
Execute o comando para iniciar o treinamento com YOLOv5:

````bash
!python yolov5/train.py --img 640 --batch 16 --epochs 10 --data coco.yaml --weights yolov5s.pt --cache
````
### 5. Visualizando as Métricas
Durante o treinamento, é possível monitorar as métricas (como `loss` e ``mAP@0.5``) com TensorBoard ou capturando os logs do treinamento. Caso queira gerar gráficos diretamente em Python a partir dos logs, utilize o código abaixo:

````python

# Exemplo para visualizar os gráficos de perda e precisão
import matplotlib.pyplot as plt
import pandas as pd

history = pd.read_csv('runs/train/exp/results.csv')  # Ou 'exp/results.json'

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plotando a perda de validação
axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss', color='red')
axes[0].set_title('Perda durante o treinamento')
axes[0].set_xlabel('Épocas')
axes[0].set_ylabel('Perda')
axes[0].legend()

# Plotando o mAP (mean Average Precision)
axes[1].plot(history['epoch'], history['val_mAP_0.5'], label='Val mAP@0.5', color='blue')
axes[1].set_title('mAP durante o treinamento')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('mAP@0.5')
axes[1].legend()

plt.tight_layout()
plt.show()
````
### 6. Resultados
Após o treinamento, o modelo será salvo na pasta ``runs/train/exp/`` (ou uma pasta com nome semelhante). Você pode usar este modelo treinado para realizar a detecção de objetos em novas imagens ou vídeos.

````python
# Exemplo de código para usar o modelo treinado para inferência
from yolov5 import YOLOv5

# Carregar o modelo treinado
model = YOLOv5('runs/train/exp/weights/best.pt')

# Realizar a detecção em uma nova imagem
results = model.predict('path/to/test/image.jpg')

# Mostrar os resultados
results.show()  # Exibe a imagem com as caixas delimitadoras
````
# Contribuições
Sinta-se à vontade para contribuir com melhorias neste repositório. Se encontrar problemas ou tiver sugestões de melhorias, crie uma issue ou faça um pull request.
