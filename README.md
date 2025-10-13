# Detecção de Tráfego WebRTC em Ambientes Criptografados

Este repositório contém o código, dados e documentação do projeto desenvolvido na disciplina de Reconhecimento de Padrões, cujo objetivo é identificar fluxos de tráfego WebRTC em meio a conexões criptografadas, utilizando técnicas clássicas de aprendizado de máquina.

## 1. Objetivo

Com o avanço da criptografia em protocolos de comunicação, como HTTPS, DTLS e SRTP, tornou-se difícil distinguir aplicações de rede por meio de inspeção direta de pacotes.  
O projeto propõe uma abordagem baseada em reconhecimento de padrões, analisando características estatísticas e temporais dos pacotes (metadados) para classificar se um fluxo pertence ou não a uma comunicação WebRTC.

## 2. Estrutura do Repositório

webrtc-traffic-detection/
├── data/ # Dados brutos e processados
│ ├── raw/ # Capturas originais (ex: ISCX VPN-nonVPN, pcap)
│ ├── processed/ # Dados convertidos pelo CICFlowMeter
│ └── readme.md
│
├── notebooks/ # Notebooks de análise e experimentação
│ ├── 01_exploracao_dataset.ipynb
│ ├── 02_treinamento_modelos.ipynb
│ ├── 03_visualizacao_resultados.ipynb
│ └── 04_captura_webrtc_local.ipynb
│
├── scripts/ # Scripts Python reutilizáveis
│ ├── preprocess_dataset.py
│ ├── train_models.py
│ └── feature_extraction_webrtc.py
│
├── reports/ # Resultados e figuras
│ ├── figures/
│ └── webrtc_traffic_detection_report.pdf
│
├── docs/ # Documentos de apoio e proposta do projeto
│ ├── proposal_overleaf.tex
│ ├── methodology_notes.md
│ └── references.bib
│
├── requirements.txt # Dependências do projeto
├── .gitignore
└── README.md


## 3. Metodologia

1. **Coleta e preparação dos dados**  
   - Utilização do dataset público [ISCX VPN-nonVPN](https://www.unb.ca/cic/datasets/vpn.html).  
   - Capturas complementares de tráfego WebRTC realizadas com Wireshark em chamadas via Google Meet, Jitsi e Asterisk.  
   - Conversão de arquivos `.pcap` em `.csv` com o uso do CICFlowMeter.

2. **Pré-processamento**  
   - Limpeza de atributos ausentes e normalização z-score.  
   - Seleção de atributos relevantes, como tamanho médio dos pacotes, duração do fluxo e tempo médio entre envios.

3. **Modelagem e classificação**  
   - Algoritmos utilizados: Support Vector Machine (SVM), Random Forest, K-Means e Isolation Forest.  
   - Avaliação por meio de métricas como Acurácia, F1-Score, Matriz de Confusão e AUC-ROC.

4. **Interpretação dos resultados**  
   - Análise da importância das variáveis.  
   - Verificação da separabilidade dos fluxos WebRTC em relação a outros protocolos criptografados.

## 4. Resultados Esperados

Espera-se que o modelo seja capaz de identificar fluxos WebRTC mesmo sob criptografia total, com desempenho estável e interpretável.  
O estudo visa contribuir para pesquisas em monitoramento de rede, QoS e segurança de comunicações multimídia.

## 5. Execução

**Requisitos mínimos:**
- Python 3.9 ou superior  
- pip e virtualenv instalados  

**Instalação:**
```bash
git clone https://github.com/leonardorpereira/webrtc-traffic-detection.git
cd webrtc-traffic-detection

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

Após a configuração, os experimentos podem ser executados a partir dos notebooks numerados em notebooks/.

6. Referências
Draper-Gil, G. et al. Characterization of Encrypted and VPN Traffic using Time-related Features. ICISSP, 2016.

Taylor, V.F. et al. Robust Identification of Encrypted Video Streams in the Wild. IMC, 2017.

Canadian Institute for Cybersecurity – ISCX VPN-nonVPN Dataset. https://www.unb.ca/cic/datasets/vpn.html
