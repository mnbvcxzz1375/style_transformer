metanet.load_state_dict(torch.load('D:\\Anaconda3Project\\styletransfer\\model\\metanet_base16_style50_tv1e-06_tagnohvd_6.pth'))
transform_net.load_state_dict(torch.load('D:\\Anaconda3Project\\styletransfer\\model\\metanet_base16_style50_tv1e-06_tagnohvd_transform_net_6.pth'))
content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
# while content_images.min() < -2:
#     print('.', end=' ')
#     content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
transformed_images = transform_net(content_images)

transformed_images_vis = torch.cat([x for x in transformed_images], dim=-1)
content_images_vis = torch.cat([x for x in content_images], dim=-1)


plt.figure(figsize=(20, 12))
plt.subplot(3, 1, 1)
imshow(style_image)
plt.subplot(3, 1, 2)
imshow(content_images_vis)
plt.subplot(3, 1, 3)
imshow(transformed_images_vis)