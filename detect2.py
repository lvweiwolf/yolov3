import argparse

from numpy.lib.npyio import savez_compressed

from models import *
from utils.datasets import *
from utils.utils import *
from utils.torch_utils import py_nms

def model_predict(img, model, half=False):
    # pred -- > [bs, 3(num_anchor) * 13 * 13, 5 + num_classes] 
    pred = model(img, augment=opt.augment)[0]
    t2 = torch_utils.time_synchronized()
    
    # 预测结果应用FLT32
    if half:
        pred = pred.float()
    
    # 应用NMS
    pred = non_max_suppression(pred, 
                                opt.conf_thres, 
                                opt.iou_thres,
                                multi_label=False,
                                classes=opt.classes,
                                agnostic=opt.agnostic_nms)
    return pred

def pad_resize(img_ori, img_size):
    # Padded resize
    img = letterbox(img_ori, new_shape=img_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    return img
   

def detect2():
    # 预测时的输入尺寸
    imgsz = opt.img_size
    source = opt.source     # 预测数据路径
    out = opt.output           # 输出路径
    weights = opt.weights   # 预测模型权重
    half = opt.half         # 是否使用F16, 默认F32
    # view_img = opt.view_img # 是否显示预测结果
    # save_txt = opt.save_txt # 保存预测结果到txt
    
    # 预测时使用cpu还是gpu
    device = torch_utils.select_device(device=opt.device)
    # 创建输出目录
    if os.path.exists(out):
        shutil.rmtree(out)
    
    os.makedirs(out)
    
    # 初始化网络模型
    model = Darknet(opt.cfg, imgsz)
    # 加载权重文件
    weights_data = torch.load(weights, map_location=device)
    # 提取模型权重，其他的是训练时的参数信息
    model.load_state_dict(weights_data['model'])
    # pytorch设置模型未评估模式
    model.to(device).eval()
    # 半精度预测，只支持CUDA
    half = half and device.type != 'cpu'
    if half:
        model.half()
        
    # 设置数据加载
    dataset = LoadImages(source, img_size=imgsz)
    # 获得类别列表、颜色列表
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    # TEST: 测试运行
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    # _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None
    
    # 兴趣区域列表
    Attention_List = [
        'Attention_GT',
        'Attention_TQ',
        'Attention_JYZ'
    ]
    
    t0 = time.time()    
    # 遍历加载数据, 格式：(file path， pad resize img data, orignal data, video capture)
    for path, img, img_ori, vic_cap in dataset:
        h_ori, w_ori = img_ori.shape[:2]
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0    # 0-255 到 0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # [1, 3, h, w]
        
        # 前向传播, # pred [1, 3xsmall_size + 3xmedium_size + 3xlarge_size, 5 + num_classes]
        t1 = torch_utils.time_synchronized()
        pred = model_predict(img, model, half)
        t2 = torch_utils.time_synchronized()
        # 打印消耗时间 (inference + NMS)
        print('%s Done. (%.3fs)' % (path, t2-t1))
        
        final_predict_dict = {
            i : {
                'final_boxes': [],
                'final_scores': []
            }
            for i in range(len(names))
        }
        
        dets = pred[0]
        if dets is None:
             continue
        
        dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], img_ori.shape).round()
        for i, det in enumerate(dets):
            # 将box坐标恢复到原始尺寸
            box = det[:4].tolist()
            score = det[4].item()
            label = int(det[5].item())
            
            x1, y1, x2, y2 = box
            labelname = names[label]
            # 左下角、右上角
            xmin = int(max(min(x1, x2), 0))
            xmax = int(min(max(x1, x2), w_ori))
            ymin = int(max(min(y1, y2), 0))
            ymax = int(min(max(y1, y2), h_ori))
            
            # 如果是Attention_GT 、 Attention_TQ, Attention_JYZ 将兴趣区域从原图裁剪进行预测
            if labelname in set(Attention_List):
                # 从原图裁剪图像
                att_img = img_ori[ymin:ymax, xmin:xmax, :]
                h_att, w_att = att_img.shape[:2]
                # 从原图裁剪出来的图像缩放到预测输入尺寸
                att_img_resize = pad_resize(att_img, imgsz)
                att_img_resize = torch.from_numpy(att_img_resize).to(device)
                att_img_resize = att_img_resize.half() if half else att_img_resize.float()
                att_img_resize /= 255.0    # 0-255 到 0.0-1.0
                if att_img_resize.ndimension() == 3:
                    att_img_resize = att_img_resize.unsqueeze(0) # [1, 3, h, w]

                t1 = torch_utils.time_synchronized()
                att_pred = model_predict(att_img_resize, model, half)
                t2 = torch_utils.time_synchronized()
                print('Attention predict Done. (%.3fs)' % (t2-t1))
                
                att_dets = pred[0]
                if att_dets is None:
                    continue
                
                att_dets[:, :4] = scale_coords(att_img_resize.shape[2:], att_dets[:, :4], att_img.shape).round()
                for j, att_det in enumerate(att_dets):
                    small_box = att_det[:4].tolist()
                    small_score = att_det[4].item()
                    small_label = int(att_det[5].item())
                    
                    x1, y1, x2, y2 = small_box
                    small_labelname = names[small_label]
                    
                    # 排除兴趣区域，我们只关心常规类别
                    if small_labelname in set(Attention_List):
                        continue
                    
                    # bbox在原图上的坐标
                    final_predict_dict[small_label]['final_boxes'].append([x1 + xmin, y1 + ymin, x2 + xmin, y2 + ymin])
                    final_predict_dict[small_label]['final_scores'].append(small_score)
            else:
                final_predict_dict[label]['final_boxes'].append([xmin, ymin, xmax, ymax])
                final_predict_dict[label]['final_scores'].append(score)

        
        # 最后进行最大化抑制，过滤过分重叠的预测框
        for i in final_predict_dict.keys():
            final_boxes_ = np.array(final_predict_dict[i]['final_boxes'])
            final_scores_ = np.array(final_predict_dict[i]['final_scores'])
            
            if len(final_boxes_) == 0:
                continue
            
            # indices = py_nms(final_boxes_, 
            #                  final_scores_, 
            #                  max_boxes=50, 
            #                  iou_thresh=opt.iou_thres)
            indices = torchvision.ops.boxes.nms(torch.from_numpy(final_boxes_).float(),
                                                torch.from_numpy(final_scores_).float(), 
                                                opt.iou_thres).tolist()
            
            final_boxes_ = final_boxes_[indices]
            final_scores_ = final_scores_[indices]
            
            for k in range(len(final_boxes_)):
                x0, y0, x1, y1 = final_boxes_[k]
                plot_one_box([x0, y0, x1, y1],
                             img_ori, 
                             label=names[i] + ', {:.2f}%'.format(final_scores_[k] * 100),
                             color=colors[i],
                             line_thickness=3)
        
        print('{} 预测时间 {:.2f}ms'.format(path, (time.time() - t0) * 1000))
        
        save_path = str(Path(out) / Path(path).name)
        imparams = [cv2.IMWRITE_JPEG_QUALITY, 82]
        cv2.imencode('.jpg', img_ori, imparams)[1].tofile(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg 网络配置文件')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names 类别文件')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='模型权重文件')
    parser.add_argument('--source', type=str, default='data/samples', help='预测数据目录')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='预测输出目录')  # output folder
    parser.add_argument('--img_size', type=int, default=512, help='预测输入尺寸(像素)')
    parser.add_argument('--conf_thres', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='NMS时的 IOU 阈值')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='输出视频编码 (确认 ffmpeg 是否支持)')
    parser.add_argument('--half', action='store_true', help='FP16 半精度预测')
    parser.add_argument('--device', default='', help='设备 id (i.e. 0 或 0,1) 或 cpu')
    parser.add_argument('--view_img', action='store_true', help='是否显示预测结果')
    parser.add_argument('--save_txt', action='store_true', help='保存预测结果到 *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='类别筛选器')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='预测时是否对数据进行增强')
    
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)
    opt.names = check_file(opt.names)
    
    print(opt)
    
    with torch.no_grad():
        detect2()