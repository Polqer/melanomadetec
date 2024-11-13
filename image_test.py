from PIL import Image
import torch
from torchvision import transforms
from step_1 import EnsembleModel, class_num, device, class_names
# Пути к модели и изображению
image_path = 'path_to_your_image.jpg'  # Искомое изображение
checkpoint_path = '/content/drive/MyDrive/Results/ensemble_model_2.ckpt'

# Загрузим модель и ее веса
model_ft = EnsembleModel(class_num)
model_ft.load_state_dict(torch.load(checkpoint_path))
model_ft = model_ft.to(device)
model_ft.eval()  # Переводим модель в режим оценки

# Преобразования для изображения
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка и преобразование изображения
image = Image.open(image_path).convert('RGB')
image_tensor = data_transforms(image).unsqueeze(0).to(device)  # Добавляем batch dimension

# Прогоняем изображение через модель
with torch.no_grad():
    outputs = model_ft(image_tensor)
    _, preds = torch.max(outputs, 1)

# Интерпретируем результат
predicted_class = preds.item()
print(f"Предсказанный класс: {class_names[predicted_class]}")
