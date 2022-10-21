def SID_model(rec_xyz_tensor, test_id):
    model = SeeInDark()
    model.load_state_dict(torch.load(
        m_path + m_name, map_location=torch.device('cuda')))
    model = model.to(device)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            _, in_fn = os.path.split(in_path)
            # print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure/in_exposure, 300)

            ###
            raw = rawpy.imread(in_path)
            ###

            im = raw.raw_image_visible.astype(np.float32)
            input_full = np.expand_dims(pack_raw(im), axis=0) * ratio

            # check main_batch.py for this function
            # im = raw.postprocess(use_camera_wb=True, half_size=False,
            # no_auto_bright=True)
            im = raw.postprocess(use_camera_wb=True, half_size=False,
                                 no_auto_bright=True, output_bps=16)
            scale_full = np.expand_dims(np.float32(im/65535.0), axis=0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True)
            gt_full = np.expand_dims(np.float32(im/65535.0), axis=0)

            input_full = np.minimum(input_full, 1.0)

            in_img = torch.from_numpy(
                input_full).permute(0, 3, 1, 2).to(device)

            ###
            with torch.no_grad():
                # rec_xyz_tensor = rec_xyz_tensor.permute(0, 1, 2).to(device)
                # rec_xyz_tensor = torch.unsqueeze(rec_xyz_tensor, dim=0)
                print(rec_xyz_tensor.shape)
                out_img = model(rec_xyz_tensor)
            ###

            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()

            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            gt_full = gt_full[0, :, :, :]
            scale_full = scale_full[0, :, :, :]
            origin_full = scale_full
            # scale the low-light image to the same mean of the groundtruth
            scale_full = scale_full*np.mean(gt_full)/np.mean(scale_full)

            Image.fromarray((origin_full*255).astype('uint8')
                            ).save(result_dir + '%5d_00_%d_ori.png' % (test_id, ratio))
            Image.fromarray((output*255).astype('uint8')
                            ).save(result_dir + '%5d_00_%d_out.png' % (test_id, ratio))
            Image.fromarray((scale_full*255).astype('uint8')
                            ).save(result_dir + '%5d_00_%d_scale.png' % (test_id, ratio))
            Image.fromarray((gt_full*255).astype('uint8')
                            ).save(result_dir + '%5d_00_%d_gt.png' % (test_id, ratio))
