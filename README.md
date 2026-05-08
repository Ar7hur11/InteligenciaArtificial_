# Projeto IA — Classificação de Imagens com YOLO

Trabalho acadêmico de Inteligência Artificial.

## Integrantes
- Arthur Campos Rezende
- Caio Fontes
- Danyelle
- João Vitor
- Gustavo falcão 
- Alberto Branco 

## Sobre o projeto
Inteligência artificial modelada e treinada para identificar letras e números de placas de carros na rua por meio de fotos.

## Como rodar

### 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

### 2. Criar ambiente virtual e instalar dependências
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3. Adicionar os dados
Coloque as imagens em `data/raw/` e configure `data.yaml` com suas classes.

### 4. Treinar
```bash
python src/train.py
```

### 5. Avaliar
```bash
python src/evaluate.py
```

### 6. Prever em nova imagem
```bash
python src/predict.py caminho/para/imagem.jpg
```

## Dataset
> Descreva aqui de onde vêm os dados e como baixá-los.

## Estrutura do projeto
```
projeto-ia/
├── data/
│   ├── raw/              ← imagens originais (não versionadas)
│   ├── processed/images/ ← imagens organizadas por split
│   └── labels/           ← anotações YOLO (.txt)
├── notebooks/            ← exploração e experimentos
├── src/                  ← código principal
├── models/               ← modelos salvos (não versionados)
├── tests/                ← testes automatizados
├── data.yaml             ← configuração do dataset YOLO
└── requirements.txt      ← dependências
```
