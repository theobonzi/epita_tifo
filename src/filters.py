import numpy as np
import pyopencl as cl
import cv2

def apply_gray(image):
    if image.ndim > 2 and image.shape[2] == 3:
        gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        gray_image = gray_image.astype(np.uint8)
        return gray_image
    else:
        return image

def apply_pixelize(image, block_size):
    height, width = image.shape[:2]
    pixelized_image = np.zeros_like(image)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            block_avg = np.mean(block, axis=(0, 1), dtype=np.float32)
            if image.ndim == 2:
                pixelized_image[i:i+block_size, j:j+block_size] = block_avg.astype(np.uint8)
            else:
                pixelized_image[i:i+block_size, j:j+block_size, :] = block_avg.astype(np.uint8)

    return pixelized_image

def gray_to_rgb(image):
    rgb_image = np.stack((image,) * 3, axis=-1)
    return rgb_image

def apply_binary(image, threshold):
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
    return binary_image

def apply_negative(image):
    negative_image = 255 - image
    return negative_image

def apply_histogram_equalization(image):
    if image.ndim > 2:
        gray_image = apply_gray(image)
    else:
        gray_image = image

    histogram, _ = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])

    cumulative_histogram = np.cumsum(histogram)

    normalized_cumulative_histogram = cumulative_histogram / float(cumulative_histogram[-1])

    equalized_image = np.interp(gray_image.flatten(), np.arange(0, 256), normalized_cumulative_histogram * 255)

    equalized_image = equalized_image.reshape(gray_image.shape).astype(np.uint8)

    return equalized_image

def apply_sobel(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    image = apply_gray(image)
    
    gradient_x = np.abs(np.apply_along_axis(lambda x: np.convolve(x, sobel_x.flatten(), mode='same'), axis=1, arr=image))
    gradient_y = np.abs(np.apply_along_axis(lambda x: np.convolve(x, sobel_y.flatten(), mode='same'), axis=0, arr=image))

    sobel_image = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    sobel_image *= 255.0 / np.max(sobel_image)

    return sobel_image.astype(np.uint8)

def apply_prewitt(image):
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    if image.ndim == 3:
        image = apply_gray(image)

    gradient_x = np.abs(np.apply_along_axis(lambda x: np.convolve(x, prewitt_x.flatten(), mode='same'), axis=1, arr=image))
    gradient_y = np.abs(np.apply_along_axis(lambda x: np.convolve(x, prewitt_y.flatten(), mode='same'), axis=0, arr=image))

    prewitt_image = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    prewitt_image *= 255.0 / np.max(prewitt_image)  # Scale result to 0-255

    return prewitt_image.astype(np.uint8)

def apply_laplace(image):
    laplace_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)

    if image.ndim == 3:
        image = apply_gray(image)

    laplace_image = np.abs(np.apply_along_axis(lambda x: np.convolve(x, laplace_kernel.flatten(), mode='same'), axis=1, arr=image))
    laplace_image *= 255.0 / np.max(laplace_image)  # Scale result to 0-255

    return laplace_image.astype(np.uint8)

def blur_image_with_opencl(img):
    platform = cl.get_platforms()[0]    
    device = platform.get_devices()[0]    
    context = cl.Context([device])    
    queue = cl.CommandQueue(context)    

    img = apply_gray(img)
    img = img.astype(np.float32)
    img /= 255.0
    img = np.ascontiguousarray(img)

    mf = cl.mem_flags
    buf_img = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    buf_out_img = cl.Buffer(context, mf.WRITE_ONLY, img.nbytes)

    prg = cl.Program(context, """
    __kernel void blur_image(__global const float4* img, __global float4* out_img, const int width, const int height)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        
        if(x > 1 && x < width-2 && y > 1 && y < height-2) {
            for(int i=-2; i<=2; i++){
                for(int j=-2; j<=2; j++){
                    sum += img[(y+j)*width + (x+i)];
                }
            }
            out_img[y*width+x] = sum / 25.0f;
        }
        else {
            out_img[y*width+x] = img[y*width+x];
        }
    }
    """).build()

    width, height = img.shape
    
    prg.blur_image(queue, (width, height), None, buf_img, buf_out_img, np.int32(width), np.int32(height))

    blur_img = np.empty_like(img)
    cl.enqueue_copy(queue, blur_img, buf_out_img)

    blur_img *= 255.0
    blur_img = cv2.convertScaleAbs(blur_img)

    return blur_img

def erode_image_with_opencl(img):
    platform = cl.get_platforms()[0]    
    device = platform.get_devices()[0]    
    context = cl.Context([device])    
    queue = cl.CommandQueue(context)    

    img = apply_gray(img)
    img = img.astype(np.float32)
    img /= 255.0
    img = np.ascontiguousarray(img)

    mf = cl.mem_flags
    buf_img = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    buf_out_img = cl.Buffer(context, mf.WRITE_ONLY, img.nbytes)

    prg = cl.Program(context, """
    __kernel void erode_image(__global const float4* img, __global float4* out_img, const int width, const int height)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        float4 min_pixel = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
        
        if(x > 0 && x < width-1 && y > 0 && y < height-1) {
            for(int i=-1; i<=1; i++){
                for(int j=-1; j<=1; j++){
                    min_pixel = min(min_pixel, img[(y+j)*width + (x+i)]);
                }
            }
            out_img[y*width+x] = min_pixel;
        }
        else {
            out_img[y*width+x] = img[y*width+x];
        }
    }
    """).build()

    width, height = img.shape
    
    prg.erode_image(queue, (width, height), None, buf_img, buf_out_img, np.int32(width), np.int32(height))

    eroded_img = np.empty_like(img)
    cl.enqueue_copy(queue, eroded_img, buf_out_img)

    eroded_img *= 255.0
    eroded_img = cv2.convertScaleAbs(eroded_img)

    return eroded_img

def dilate_image_with_opencl(img):
    platform = cl.get_platforms()[0]    
    device = platform.get_devices()[0]    
    context = cl.Context([device])    
    queue = cl.CommandQueue(context)    

    img = apply_gray(img)
    img = img.astype(np.float32)
    img /= 255.0
    img = np.ascontiguousarray(img)

    mf = cl.mem_flags
    buf_img = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    buf_out_img = cl.Buffer(context, mf.WRITE_ONLY, img.nbytes)

    prg = cl.Program(context, """
    __kernel void dilate_image(__global const float4* img, __global float4* out_img, const int width, const int height)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        float4 max_pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        
        if(x > 0 && x < width-1 && y > 0 && y < height-1) {
            for(int i=-1; i<=1; i++){
                for(int j=-1; j<=1; j++){
                    max_pixel = max(max_pixel, img[(y+j)*width + (x+i)]);
                }
            }
            out_img[y*width+x] = max_pixel;
        }
        else {
            out_img[y*width+x] = img[y*width+x];
        }
    }
    """).build()

    width, height = img.shape
    
    prg.dilate_image(queue, (width, height), None, buf_img, buf_out_img, np.int32(width), np.int32(height))

    dilated_img = np.empty_like(img)
    cl.enqueue_copy(queue, dilated_img, buf_out_img)

    dilated_img *= 255.0
    dilated_img = cv2.convertScaleAbs(dilated_img)

    return dilated_img