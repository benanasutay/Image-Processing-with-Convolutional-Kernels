import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Konvolüsyon fonksiyonu
def convolution(image, kernel):
    image_rows, image_cols = image.shape
    kernel_rows, kernel_cols = kernel.shape
    
    # Görüntüyü pad et (kenar bölgeler için)
    pad_height = kernel_rows // 2
    pad_width = kernel_cols // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # Çıkış görüntüsü
    output = np.zeros_like(image)
    
    # Konvolüsyon işlemi
    for i in range(image_rows):
        for j in range(image_cols):
            region = padded_image[i:i + kernel_rows, j:j + kernel_cols]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Lenna resmini yükle ve gri tonlamalıya dönüştür
try:
    image = Image.open(r'C:\Users\BENAN\Desktop\HW2- Tek çıktıda\Lenna.png').convert('L')  # Eğer Lenna resmini farklı bir yol ile kullanıyorsanız, bu kısmı değiştirin
    image = np.array(image)
except Exception as e:
    print(f"Image loading failed: {e}")

# PDF'te verilen kernel'ları tanımla
kernel_edge1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
kernel_edge2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
kernel_box_blur = (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
kernel_gaussian_blur_3x3 = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
kernel_gaussian_blur_5x5 = (1/256) * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])

# Konvolüsyon işlemini uygula
result_edge1 = convolution(image, kernel_edge1)
result_edge2 = convolution(image, kernel_edge2)
result_sharpen = convolution(image, kernel_sharpen)
result_box_blur = convolution(image, kernel_box_blur)
result_gaussian_blur_3x3 = convolution(image, kernel_gaussian_blur_3x3)
result_gaussian_blur_5x5 = convolution(image, kernel_gaussian_blur_5x5)

# Görüntüleri normalize et
def normalize_image(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # 0-1 aralığına normalizasyon
    img = (img * 255).astype(np.uint8)  # 0-255 aralığına ölçekleme
    return img

# Normalize edilmiş görüntüleri görselleştir
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Orijinal Görüntü')

plt.subplot(2, 3, 2)
plt.imshow(normalize_image(result_edge1), cmap='gray')
plt.title('Kenar Belirleme 1')

plt.subplot(2, 3, 3)
plt.imshow(normalize_image(result_edge2), cmap='gray')
plt.title('Kenar Belirleme 2')

plt.subplot(2, 3, 4)
plt.imshow(normalize_image(result_sharpen), cmap='gray')
plt.title('Keskinleştirme')

plt.subplot(2, 3, 5)
plt.imshow(normalize_image(result_box_blur), cmap='gray')
plt.title('Kutu Bulanıklaştırma')

plt.subplot(2, 3, 6)
plt.imshow(normalize_image(result_gaussian_blur_3x3), cmap='gray')
plt.title('Gaussian Bulanıklaştırma 3x3')

plt.tight_layout()
plt.show()
