import  cv2, os, audio, time, pygame, math, datetime, subprocess
import  requests
from    tqdm import tqdm
import  numpy as np

import  onnxruntime

import 	torch
import  torch.nn.functional as F
import  torch.multiprocessing as mp
from 	models import Wav2Lip
from 	insightface_func.face_detect_crop_single import Face_detect_crop


def cal_torch_theta(opencv_theta: np.ndarray, src_h: int, src_w: int, dst_h: int, dst_w: int):
    m = np.concatenate([opencv_theta, np.array([[0., 0., 1.]], dtype=np.float32)])
    m_inv = np.linalg.inv(m)

    a = np.array([[2 / (src_w - 1), 0., -1.],
                  [0., 2 / (src_h - 1), -1.],
                  [0., 0., 1.]], dtype=np.float32)

    b = np.array([[2 / (dst_w - 1), 0., -1.],
                  [0., 2 / (dst_h - 1), -1.],
                  [0., 0., 1.]], dtype=np.float32)
    b_inv = np.linalg.inv(b)

    pytorch_m = a @ m_inv @ b_inv
    return torch.as_tensor(pytorch_m[:2], dtype=torch.float32)
    # return pytorch_m

class InteractiveLip:
    def __init__(self,
                    fps=25,
                    pads=[0, 20, 0, 0],
                    device="cuda:0",
                    database="./database",
                    img_size=96,
                    batch_size=1,
                    crop_size=512,
                    mel_step_size=16,
                    resize_factor=2.0,
                    audio_tmp_dir="./audio_temp") -> None:
        self.fps            = fps
        self.pads           = pads
        self.device         = device
        self.database       = database
        self.resize_factor  = resize_factor
        self.crop_size 	    = crop_size
        self.img_size       = img_size
        self.batch_size     = batch_size
        self.mel_step_size  = mel_step_size
        self.audio_tmp_dir  = audio_tmp_dir
        self.mel_idx_multiplier = 80./self.fps
        self.detect         = Face_detect_crop(name='antelope', root='./insightface_func/models')
        self.detect.prepare(ctx_id = 0, det_thresh=0.6,\
                                det_size=(640,640),mode = "ffhq")
        os.makedirs("temporal", exist_ok=True)
        os.makedirs(audio_tmp_dir, exist_ok=True)

        from modelscope.pipelines import pipeline as asr_pipeline
        from modelscope.utils.constant import Tasks
        self.asr_pipeline   = asr_pipeline(
                                task=Tasks.auto_speech_recognition,
                                model='damo/speech_paraformer-large'+
                                    '_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                                device="gpu"
                            )
        from TTS.synthesize_all import SpeechSynthesis
        self.tts_pipeline   = SpeechSynthesis('./TTS/config/AISHELL3')
    
    def make_resources(self, video):
        if not os.path.isfile(video):
            raise ValueError('--face argument must be a valid path to video/image file')
        basepath    = os.path.splitext(os.path.basename(video))[0]
        tg_path     = os.path.join(self.database, basepath)
        if os.path.exists(os.path.join(tg_path,"ffhq_list.pth")):
            print("Resource already exists!")
            self.load_resources(video)
            return

        
        os.makedirs(tg_path, exist_ok=True)
        
        video_stream = cv2.VideoCapture(video)
        self.fps     = video_stream.get(cv2.CAP_PROP_FPS)
        print('Reading video frames...')
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if self.resize_factor > 1:
                frame = cv2.resize(frame, (int(frame.shape[1]//self.resize_factor), 
                                           int(frame.shape[0]//self.resize_factor)))
            full_frames.append(frame)

        crop_size 	    = self.crop_size
        kernel_size     = (20, 20)
        blur_size       = tuple(2*i+1 for i in kernel_size)
        kernel          = np.ones((40,40),np.uint8)
        pady1, pady2, padx1, padx2 = self.pads
        img_list  		= []
        mat_list  		= []
        mat_inv_list 	= []
        bbox_list 		= []
        ffhq_list 		= []
        imgmask_list 	= []
        self.frame_h, self.frame_w= full_frames[0].shape[:-1]


        for img_i in tqdm(full_frames[:250]):
            ffhq_img, mat, mat_inv 	= self.detect.get_invmat(img_i, crop_size)
            ffhq_img 				= cv2.copyMakeBorder(ffhq_img, 100, 
                                        100, 100, 100, cv2.BORDER_CONSTANT, value=(0,0,0))
            bboxes 					= self.detect.det_model.detect(ffhq_img,
                                                threshold=self.detect.det_thresh,
                                                max_num=0,
                                                metric='default')[0][0]
            bboxes = [int(x) for x in bboxes[:4]]
            y1 = max(0, bboxes[1] - pady1)
            y2 = min(ffhq_img.shape[0], bboxes[3] + pady2)
            x1 = max(0, bboxes[0] - padx1)
            x2 = min(ffhq_img.shape[1], bboxes[2] + padx2)
            ffhq_img= ffhq_img/255.0
            face 	= ffhq_img[y1: y2, x1:x2]
            face 	= cv2.resize(face, (self.img_size, self.img_size))
            face_masked = face.copy()
            face_masked[self.img_size//2:,:] = 0
            face 	= np.concatenate((face_masked, face), axis=-1)
            ffhq_img= 2 * (ffhq_img[:,:,[2,1,0]] - 0.5)
            ffhq_list.append(ffhq_img)
            img_list.append(face)
            mat_list.append(mat)
            torch_theta = cal_torch_theta(mat_inv, crop_size, crop_size, self.frame_h, self.frame_w).unsqueeze(0)
            grid = F.affine_grid(torch_theta, size=[1, 3, self.frame_h, self.frame_w])
            mat_inv_list.append(grid)

            bbox_list.append([x1,x2,y1,y2])

            img_mask   = np.full((crop_size,crop_size), 255, dtype=float)
            img_mask   = cv2.warpAffine(img_mask, mat_inv, (self.frame_w, self.frame_h))
            img_mask[img_mask>20] =255
            img_mask    = cv2.erode(img_mask, kernel, iterations = 1)
            img_mask    = cv2.GaussianBlur(img_mask, blur_size, 0)
            img_mask    /= 255
            img_mask    = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            imgmask_list.append(img_mask)
        img_list    = np.asarray(img_list)
        img_list    = torch.FloatTensor(np.transpose(img_list, (0, 3, 1, 2)))
        ffhq_list	= np.asarray(ffhq_list)
        ffhq_list	= torch.FloatTensor(np.transpose(ffhq_list, (0, 3, 1, 2)))

        full_frames	= np.asarray(full_frames)
        full_frames	= torch.FloatTensor(np.transpose(full_frames, (0, 3, 1, 2)))

        imgmask_list= np.asarray(imgmask_list)
        imgmask_list= torch.FloatTensor(np.transpose(imgmask_list, (0, 3, 1, 2)))

        # mat_inv_list	= np.asarray(mat_inv_list)
        mat_inv_list	= torch.cat(mat_inv_list, dim=0)

        self.img_list  		= img_list
        self.mat_list  		= mat_list
        self.mat_inv_list 	= mat_inv_list
        self.bbox_list 		= bbox_list
        self.ffhq_list 		= ffhq_list
        self.imgmask_list 	= imgmask_list
        self.full_frames    = full_frames[:,[2,1,0],:,:]

        torch.save(self.img_list , os.path.join(tg_path,"img_list.pth"))
        torch.save(self.mat_list , os.path.join(tg_path,"mat_list.pth"))
        torch.save(self.mat_inv_list , os.path.join(tg_path,"mat_inv_list.pth"))
        torch.save(self.bbox_list , os.path.join(tg_path,"bbox_list.pth"))
        torch.save(self.ffhq_list , os.path.join(tg_path,"ffhq_list.pth"))
        torch.save(self.imgmask_list , os.path.join(tg_path,"imgmask_list.pth"))
        torch.save(self.full_frames , os.path.join(tg_path,"full_frames.pth"))
        torch.save([self.frame_h, self.frame_w], os.path.join(tg_path,"frame_size.pth"))

        self.index          = [i for i in range(self.img_list.shape[0])]
        self.index          += self.index[::-1]
        
        print("Resources saved!")

    def load_resources(self, video_path):
        basepath    = os.path.splitext(os.path.basename(video_path))[0]
        tg_path     = os.path.join(self.database, basepath)

        self.img_list  		= torch.load(os.path.join(tg_path,"img_list.pth"))
        print("img_list shape: ", self.img_list.shape)
        # self.mat_list  		= torch.load(os.path.join(tg_path,"mat_list.pth"))

        self.mat_inv_list 	= torch.load(os.path.join(tg_path,"mat_inv_list.pth"))
        print("mat_inv_list shape: ", self.mat_inv_list.shape)
        self.bbox_list 		= torch.load(os.path.join(tg_path,"bbox_list.pth"))
        self.ffhq_list 		= torch.load(os.path.join(tg_path,"ffhq_list.pth"))
        print("ffhq_list shape: ", self.ffhq_list.shape)
        self.imgmask_list 	= torch.load(os.path.join(tg_path,"imgmask_list.pth"))
        print("imgmask_list shape: ", self.imgmask_list.shape)
        self.full_frames    = torch.load(os.path.join(tg_path,"full_frames.pth"))
        print("full_frames shape: ", self.full_frames.shape)
        self.frame_h, self.frame_w=torch.load(os.path.join(tg_path,"frame_size.pth"))
        print("Resources loaded!")

        self.index          = [i for i in range(self.img_list.shape[0])]
        self.index          += self.index[::-1]

    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint
    
    def load_model(self, path):
        self.model = Wav2Lip()
        # print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        self.model.load_state_dict(new_s)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        # self.model = torch.compile(self.model, backend="inductor")

    def process_audio(self, audio_path):
        if not audio_path.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')
            subprocess.call(command, shell=True)
            audio_path = 'temp/temp.wav'
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        # print(mel.shape)
        mel_chunks = []
        i = 0
        while 1:
            start_idx = int(i * self.mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1
        
        print("Length of mel chunks: {}".format(len(mel_chunks)))
        # to batch
        mel_chunks = torch.FloatTensor(np.asarray(mel_chunks)).unsqueeze(-1)
        mel_chunks = mel_chunks.permute(0, 3, 1, 2)
        self.mel_chunks = mel_chunks

    def tensorrt_init(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path, 
                        providers=["TensorrtExecutionProvider","CUDAExecutionProvider"])
        self.io_binding = self.session.io_binding()

        # self.input_buff     = torch.empty((1,3,512,512), dtype=torch.float32, device="cuda").contiguous()
        # self.output_buff    = torch.empty((1,3,512,512), dtype=torch.float32, device="cuda").contiguous()

        # self.io_binding.bind_input(name='input',
        #                     device_type='cuda', device_id=0, 
        #                     element_type=np.float32, shape=(1,3,512,512), 
        #                     buffer_ptr=self.input_buff.data_ptr())
        # self.io_binding.bind_output(name='output',
        #                     device_type='cuda', device_id=0,
        #                     element_type=np.float32, shape=(1,3,512,512),
        #                     buffer_ptr=self.output_buff.data_ptr())
        # self.session.run_with_iobinding(self.io_binding)
        print("TensorRT backend build successful!")
    
    def generate_audio(self, audio):
        # current_time11      = datetime.datetime.now()
        asr_result_map  = self.asr_pipeline(audio)
        asr_result      = asr_result_map['text']
        
        # current_time2       = datetime.datetime.now()
        # print('ASR preprocessing time:', current_time2 - current_time11)
        # current_time11      = datetime.datetime.now()
        data            = {
                            "prompt": asr_result + ", \
                            简短回答", "history": []
                        }
        response        = requests.post("http://192.168.100.15:8000", 
                                    json=data,
                                    headers={"Content-Type": "application/json"})
        chat_result_dict= eval(response.text)
        chat_input      = chat_result_dict["response"]
        # current_time2       = datetime.datetime.now()
        # print('ChatGLM preprocessing time:', current_time2 - current_time11)
        # current_time11      = datetime.datetime.now()
        self.tts_pipeline.text2speech(chat_input, self.audio_tmp_dir)
        # current_time2       = datetime.datetime.now()
        # print('tts preprocessing time:', current_time2 - current_time11)
    
    def invoke_task(self):
        self.task_id += 1
        self.input_queue.put({
            "task_id": self.task_id
        })

    def invoke_gui_process(
                            self, 
                            threads_num =1
                        ):
        print("start to build metahuman process")
        self.task_id        = 0

        self.ctx            = mp.get_context('spawn')
        self.result_queue   = self.ctx.Queue()
        for _ in range(threads_num):
            p = self.ctx.Process(target = GUI_Task_Block, 
                                 args = (
                                        self.result_queue,
                                        self.audio_tmp_dir
                                     ))
            p.start()

    def __call__(self):
        self.process_audio(self.audio_tmp_dir + "/tmp.wav")
        multi = self.mel_chunks.shape[0]//len(self.index)
        delta = self.mel_chunks.shape[0]%len(self.index)
        index = self.index*multi + self.index[:delta]
        print("Final video lenghth %d"%len(index))

        mel_chunks = self.mel_chunks
        self.task_id += 1
        self.result_queue.put({
                            'task_id': self.task_id,
                            'chunk_num' : mel_chunks.shape[0],
                            "batch_num" : math.ceil(mel_chunks.shape[0]/self.batch_size)
                        })     
  
        input_buff     = torch.empty((1,3,512,512), dtype=torch.float32, device="cuda").contiguous()
        output_buff    = torch.empty((1,3,512,512), dtype=torch.float32, device="cuda").contiguous()
        # io_binding = tensorrt_sess.io_binding()
        for i_batch in tqdm(range(0,mel_chunks.shape[0], self.batch_size)):
            if i_batch + self.batch_size >= mel_chunks.shape[0]:
                end = mel_chunks.shape[0]
            else:
                end = i_batch+self.batch_size
            index_i   = index[i_batch:end]
            img_batch = self.img_list[index_i,:,:,:].clone().to(self.device)
            mel_batch = mel_chunks[i_batch:end,:,:,:].clone().to(self.device)
            ffhq_batch= self.ffhq_list[index_i,:,:,:].clone().to(self.device)
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)
                pred = 2 * pred[:,[2,1,0],:,:] - 1
                # pred = 2 * pred - 1
            for i in range(pred.shape[0]):
                x1, x2, y1, y2 = self.bbox_list[index[i_batch + i]]
                temp = F.interpolate(pred[[i],:,:,:],(y2 - y1, x2 - x1))
                ffhq_batch[[i],:,y1:y2, x1:x2] = temp
            ffhq_batch = ffhq_batch[:,:,100:-100,100:-100].contiguous()

            for i in range(ffhq_batch.shape[0]):
                input_buff = ffhq_batch[[i]]#.contiguous()
                self.io_binding.bind_input(name='input', device_type='cuda', 
                                        device_id=0, element_type=np.float32, 
                                        shape=(1,3,512,512), buffer_ptr=input_buff.data_ptr())
                self.io_binding.bind_output(name='output', device_type='cuda', 
                                        device_id=0, element_type=np.float32, 
                                        shape=(1,3,512,512), buffer_ptr=output_buff.data_ptr())
                self.session.run_with_iobinding(self.io_binding)
                ffhq_batch[[i]] = output_buff
            ffhq_batch = F.grid_sample(ffhq_batch, 
                        grid=self.mat_inv_list[index_i,:,:,:].to(self.device),
                        mode="bilinear", padding_mode="zeros", align_corners=False)
            ffhq_batch      = (ffhq_batch + 1.0)*127.5
            mask            = self.imgmask_list[index_i,:,:,:].to(self.device)
            full_f          = self.full_frames[index_i,:,:,:].to(self.device)
            ffhq_batch      = mask * ffhq_batch + (1-mask)*full_f
            ffhq_batch      = torch.clip(ffhq_batch, 0, 255)
            ffhq_batch      = ffhq_batch[:,[2,1,0],...].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            self.result_queue.put({
                            'task_id': self.task_id,
                            'frames' : list(ffhq_batch)
                        })


def GUI_Task_Block(
                result_queue,
                audio_tmp_dir
            ):
  
    print("GUI subprocess start successful!")
    while True:
        print("Waiting for results.......")
        chunk           = result_queue.get()
        print("Get response head!")
        pygame.init()
        sound           = pygame.mixer.Sound(audio_tmp_dir + "/tmp.wav")
        sound_length    = sound.get_length()
        pygame.mixer.music.load(audio_tmp_dir + "/tmp.wav")
        time_len        = sound_length / chunk["chunk_num"]
        frame_index     = 0
        for i_chunk in range(chunk["batch_num"]):
            frames      = result_queue.get()
            if i_chunk  == 0:
                current_time    = datetime.datetime.now()
                pygame.mixer.music.play()
            cv2.imshow('frame', frames["frames"][0])
            cv2.waitKey(1)
            other_time      = datetime.datetime.now()
            time_difference = other_time - current_time
            sleep_time      = (frame_index + 1)*time_len -\
                                time_difference.total_seconds()
            time.sleep(max(sleep_time, 0))
            frame_index += 1
        pygame.mixer.music.stop()
        pygame.quit()

if __name__ == "__main__":
    lip = InteractiveLip()
    lip.load_model("./checkpoints/wav2lip.pth")
    lip.make_resources("./inputs/ts-tim.mp4")
    lip.tensorrt_init("./GFPGANv1.4.onnx")
    lip("./inputs/libai.wav")
    lip("tmp.wav")
    # lip("jys.wav")
    # lip("jys.wav")