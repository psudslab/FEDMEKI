import torch
from model.Octavius import Octavius

from model.LAMM.conversations import conv_templates
from .test_base import TestBase


class TestOctavius(TestBase):

    def __init__(self, **args):
        super().__init__()
        self.args = args

        self.conv_mode = args['conv_mode']
        self.task_type = 'normal'
        self.max_tgt_len = args['max_tgt_len']

        delta_ckpt = torch.load(args['delta_ckpt_path'], 'cpu')
        self.model = Octavius(**args)
        
        # print(self.model.state_dict().keys())
        # print(delta_ckpt.keys())
        info = self.model.load_state_dict(delta_ckpt, strict=False)
        print(delta_ckpt.keys())
        # print(info)
        # print(delta_ckpt.keys())
        # print(self.model.state_dict().keys())
        # print(info.missing_keys)
        # print(info.unexpected_keys)
        # info = self.model.load_state_dict(delta_ckpt, strict=False)
        # info = self.model.load_state_dict(delta_ckpt)
        # delta_ckpt = torch.load(args['delta_ckpt_path'].replace(""), 'cpu')
        # print(self.model.visual_encoder)
        # print(info)
        # print(self.model.state_dict()['visual_encoder.deit.encoder.layer.0.attention.attention.query.weight'])
        # saved = torch.load("/data/xiaochen/FedMFM/ckpt/octavius_2d_e4_bs64/5/visual.pt")
        # print(saved.state_dict()['deit.encoder.layer.0.attention.attention.query.weight'])
        # print(self.model.visual_encoder['visual_encoder.deit.encoder.layer.0.attention.attention.query.weight'] == saved['visual_encoder.deit.encoder.layer.0.attention.attention.query.weight'])
        # print(self.model.visual_encoder == saved)
        # self.model.visual_encoder = saved
        self.model = self.model.eval().half()
        self.move_to_device()
        

    def move_to_device(self):
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda:0'
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)

    def generate_conversation_text(self, input_list, history, sys_msg = None):
        conv = conv_templates[self.conv_mode]
        if sys_msg:
            if conv.sys_temp is not None:
                conv.system = conv.sys_temp.format(system_message=sys_msg)
            else:
                conv.system = sys_msg
        prompts_list = []
        for input in input_list:
            prompts = ''
            prompts += conv.system + '\n\n'
            for q, a in history:
                prompts += "{}: {}\n{} {}: {}\n{}".format(conv.roles[0], q, conv.sep, conv.roles[1], a, conv.sep2 if (conv.sep2 is not None) else conv.sep)
            prompts += "{}: {}\n{}".format(conv.roles[0], input, conv.sep)
            prompts_list.append(prompts)
        return prompts_list

    @torch.no_grad()
    def do_generate(self, modality_inputs, question_list, top_p=0.9, temperature=1.0, task = None, modalities = None):
        response = self.model.generate(inputs = {
            'prompt': question_list,
            'modality_paths': modality_inputs,
            'top_p': top_p,
            'temperature': temperature,
            'max_tgt_len': self.max_tgt_len,
            'modality_embeds': []
        },
            task = task, 
            modalities = modalities)

        conv = conv_templates[self.conv_mode]
        ans_list = []
        
        for res in response:
            ans_list.append(res.split(conv.sep2 if conv.sep2 is not None else conv.sep)[0])
        # print("questions:",question_list)
        # print("answers:", ans_list)
        return ans_list

    @torch.no_grad()
    def do_generate_vqa(self, modality_inputs, question_list, top_p=0.9, temperature=1.0):
        conv = conv_templates[self.conv_mode]
        reasoning_list = self.do_generate(modality_inputs, question_list)
        option_prompt = []
        for prompt_1, response_1 in zip(question_list, reasoning_list):
            option_prompt.append(prompt_1 + response_1 + f' {conv.sep2 if conv.sep2 is not None else conv.sep}\nANSWER:')
        final_answer_list = self.do_generate(modality_inputs, option_prompt) 
        all_answer_list = []
        for reasoning, option in zip(reasoning_list, final_answer_list):
            all_answer_list.append(reasoning + '\n The answer is ' + option)
        return all_answer_list

    @torch.no_grad()
    def do_generate_3d(self, modality_inputs, question_list, top_p=0.9, temperature=1.0):
        modality_inputs.update({
            'top_p': top_p,
            'temperature': temperature,
            'max_tgt_len': self.max_tgt_len,
            'modality_embeds': [],
            'prompt': question_list,
        })
        response = self.model.generate(modality_inputs)

        conv = conv_templates[self.conv_mode]
        ans_list = []
        for res in response:
            ans_list.append(res.split(conv.sep2 if conv.sep2 is not None else conv.sep)[0])
        return ans_list

    @torch.no_grad()
    def generate(
        self, 
        modality_input, 
        question, 
        sys_msg=None, 
        dataset_name=None, 
        task_name=None,
        **kwargs
    ):
        # prompts = self.generate_conversation_text([question], history=[], sys_msg=sys_msg)
        # prompts = self.generate_conversation_text([question], history=[], sys_msg=None)
        prompts = question
        if task_name.endswith('octavius3d'):
            outputs = self.do_generate_3d(modality_input, prompts)
        else:
            if dataset_name == "ScienceQA":
                outputs = self.do_generate_vqa([modality_input], prompts)
            else:
                outputs = self.do_generate([modality_input], prompts)
        return outputs[0]

    @torch.no_grad()
    def batch_generate(
        self, 
        modality_inputs, 
        question_list, 
        sys_msg=None, 
        dataset_name=None, 
        task = None,
        modalities = None,
        **kwargs
    ):
        # prompts = self.generate_conversation_text(question_list, history=[], sys_msg=sys_msg)
        prompts = question_list
        
        if dataset_name == "ScienceQA":
            outputs = self.do_generate_vqa(modality_inputs, prompts)
        else:
            outputs = self.do_generate(modality_inputs, prompts, task = task, modalities = modalities)
        return outputs
    
    @torch.no_grad()
    def batch_generate_3d(
        self, 
        modality_inputs, 
        question_list, 
        sys_msg=None, 
        **kwargs
    ):
        prompts = self.generate_conversation_text(question_list, history=[], sys_msg=sys_msg)
        outputs = self.do_generate_3d(modality_inputs, prompts)
        return outputs