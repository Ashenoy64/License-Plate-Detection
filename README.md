# License Plate Number Detection

This project focuses on detecting license plate numbers from vehicle images using computer vision techniques.

## Installation

To get started with the project, follow these steps:

1. Clone the repository from GitHub using the following command:

   ```bash
   git clone https://github.com/Ashenoy64/License-Plate-Detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd License-Plate-Detection
   ```

3. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the model weights for license plate detection (note: specific model weights will be provided later).

## Usage

To use the License Plate Number Detection class, follow these steps:

1. Import the `LicensePlateNumberDetection` class into your Python script:

   ```python
   from LicensePlateNumberDetection import LicensePlateNumberDetection
   ```

2. Instantiate the `LicensePlateNumberDetection` class:

   ```python
   lpnd = LicensePlateNumberDetection()
   ```

3. There are two main methods available in the class:

   - `directoryInput(input_dir=directory_path)`: This method goes through the directory of images, detects license plate numbers, and returns an array of objects.

   - `detect(img, image_name)`: This method takes a CV2 image object (`img`) and the name of the image (`image_name`), and it returns the object with the detected license plate number.

## Example

```python
# Example usage
from LicensePlateNumberDetection import LicensePlateNumberDetection

# Instantiate the class
lpnd = LicensePlateNumberDetection()

# Detect license plate numbers from a directory of images
result_array = lpnd.directoryInput(input_dir='/path/to/images/')

# Detect license plate number from a single image
img = cv2.imread('/path/to/single/image.jpg')
image_name = 'image.jpg'
result_object = lpnd.detect(img, image_name)
```

## Acknowledgements

The code for this project was developed following the tutorial by Computer Vision Engineer available at: [YouTube Tutorial Link](https://youtu.be/73REqZM1Fy0).


---