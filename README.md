# App de entrenamiento para el sistema de recomendación

## La app contiene la siguiente estructura


```
.
├── Dockerfile  # Instrucciones para generar la imagen de entrenamiento.
├── README.md
├── requirements.txt # Librerías necesarias para entrenar el modelo.
└── trainer         
    └── train.py # Código de entrenamiento

```

## Funcionamiento

Para entrenar el modelo, a través de esta app, hay que generar la imagen con docker
y pushearla al container registry de GCP, luego se la llama desde el servicio
"Training" de Vertex AI. 


