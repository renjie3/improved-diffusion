import PIL.Image as Image
img = Image.open('/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/malicious2.png').convert('RGBA')
W, L = img.size
white_pixel = (255, 255, 255, 255)
for h in range(W):
    for i in range(L):
        # if img.getpixel((h, i)) == white_pixel:
        if img.getpixel((h, i))[0] > 128 and img.getpixel((h, i))[1] > 128 and img.getpixel((h, i))[2] > 128 and img.getpixel((h, i))[3] > 128:
            img.putpixel((h, i), (0, 0, 0, 0))
img.save('/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/malicious3.png')