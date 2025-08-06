from VLM import VLM

class METHODS:
    def __init__(self, model, task, method, ds, api_key=None, gpu_count=None):
        self.model = model
        self.task = task
        self.method = method
        self.api_key = api_key
        self.ds = ds

        if method in ['zero_shot', 'cocot', 'ddcot']:
            self.prompt_path = f'prompts_task{task}/{method}.txt'
            self.prompt = open(self.prompt_path, 'r').read()
        elif method == 'self_refine':
            self.prompt_path_1 = f'prompts_task{task}/self_refine_2_review.txt'
            self.prompt_path_2 = f'prompts_task{task}/self_refine_3_improve.txt'
            self.prompt_1 = open(self.prompt_path_1, 'r').read()
            self.prompt_2 = open(self.prompt_path_2, 'r').read()
        elif method == 'mad_each_debate':
            self.prompt_path_1 = f'prompts_task{task}/mad_debater_first.txt'
            self.prompt_path_2 = f'prompts_task{task}/mad_debater_second.txt'
            self.prompt_1 = open(self.prompt_path_1, 'r').read()
            self.prompt_2 = open(self.prompt_path_2, 'r').read()
        elif method == 'mad_moderate_extractive':
            self.prompt_path = f'prompts_task{task}/mad_moderator_extractive.txt'
            self.prompt = open(self.prompt_path, 'r').read()
        

        if self.task == 1:
            self.must_word = 'More effective:'
        else:
            self.must_word = ''

        if model == 'gpt-4o':
            self.vlm_model = VLM("gpt-4o", api_key)
        elif model == 'claude':
            self.vlm_model = VLM("claude", api_key)
        elif model == 'o1':
            self.vlm_model = VLM("o1", api_key)
        elif 'qwen2_5_vl' in model:
            self.vlm_model = VLM("qwen2_5_vl", api_key, gpu_count, size=model.split('_')[-1].upper())  # e.g., qwen2_5_vl_7b, qwen2_5_vl_32b
        elif 'internvl' in model:
            self.vlm_model = VLM("internvl", api_key, gpu_count, size=model.split('_')[-1].upper())  # e.g., internvl_8b, internvl_38b
        elif 'llava-next' in model:
            self.vlm_model = VLM("llava-next", api_key, gpu_count)
        elif 'llava-onevision' in model:
            self.vlm_model = VLM("llava-onevision", api_key, gpu_count)
        else:
            raise ValueError("Unsupported model name.")

    def run(self, num, first_file, second_file, start_data=None):
        if self.model in ['gpt-4o', 'claude', 'o1']:
            d2c_image = self.image_append(num, first_file, second_file)
        else:
            d2c_image = self.image_append_vllm(num, first_file, second_file)

        if self.method in ['zero_shot', 'cocot', 'ddcot']:
            return self.one_pipeline(d2c_image)
        elif self.method == 'self_refine':
            return self.self_refine(d2c_image, start_data)
        elif self.method == 'mad_each_debate':
            return self.mad_each_debate(d2c_image, start_data)
        elif self.method == 'mad_moderate_extractive':
            return self.mad_moderate_extractive(d2c_image, start_data)
        else:
            raise ValueError("Unsupported method.")
    
    def image_append(self, num, first_file, second_file):
        img_path = self.ds[num][f'{first_file}']
        img_path1 = self.ds[num][f'{second_file}']
        
        encoded_img = self.vlm_model.encode_image(img_path)
        encoded_img1 = self.vlm_model.encode_image(img_path1)

        d2c_image = []
        d2c_image.append(encoded_img)
        d2c_image.append(encoded_img1)
        return d2c_image
    
    def image_append_vllm(self, num, first_file, second_file):
        img_path = self.ds[num][f'{first_file}']
        img_path1 = self.ds[num][f'{second_file}']

        d2c_image = []
        d2c_image.append(img_path)
        d2c_image.append(img_path1)
        return d2c_image
    
    def one_pipeline(self, d2c_image):
        full_result = [[],[]]
        error = True
        error_count = 0
        while error:
            if error_count > 5:
                print("Too many errors, stopping execution.")
                return [[f'{self.must_word}'], [[]]]
            try:
                result = self.vlm_model.run(self.prompt, d2c_image, [])
                answer = result[0]
                print(answer)
                if self.must_word in answer:
                    error = False
                else:
                    error_count += 1
                    print("did not return the expected format. Retrying...")
            except Exception as e:
                error_count += 1
                print('Error:', e)
                print('Retrying...')
        full_result[0].append(answer)
        full_result[1].append(result[1:])  # tokens info
        return full_result

    def self_refine(self, d2c_image, start_data=None):
        current_step = len(start_data[0]) + 1
        if current_step == 2:
            prompt = self.prompt_1.format(previous_answer=start_data[0][0])
        else:
            prompt = self.prompt_2.format(previous_answer=start_data[0][0], feedback=start_data[0][1])
        full_result = start_data.copy()

        error = True
        error_count = 0
        while error:
            if error_count > 5:
                print("Too many errors, stopping execution.")
                full_result[1].append([])  # tokens info
                if current_step == 3:
                    full_result[0].append(f'{self.must_word}')
                else:
                    full_result[0].append('No answer')
                return full_result
            try:
                result = self.vlm_model.run(prompt, d2c_image, [])
                answer = result[0]
                print(answer)
                if current_step == 3:
                    if self.must_word in answer:
                        error = False
                    else:
                        error_count += 1
                        print("did not return the expected format. Retrying...")
                else:
                    error = False
            except Exception as e:
                error_count += 1
                print('Error:', e)
                print('Retrying...')

        full_result[0].append(answer)
        full_result[1].append(result[1:])  # tokens info
        return full_result

    def mad_each_debate(self, d2c_image, start_data=None):
        current_step = len(start_data[0]) + 1
        if current_step == 1:
            prev_opinion = ''
        else:
            prev_opinion = start_data[0][-1]

        if current_step % 2 == 1:
            prompt = self.prompt_1.format(opponent_opinion=prev_opinion)
        else:
            prompt = self.prompt_2.format(opponent_opinion=prev_opinion)
        full_result = start_data.copy()
        
        error = True
        error_count = 0
        while error:
            if error_count > 3:
                print("Too many errors, stopping execution.")
                full_result[0].append('')
                full_result[1].append([])
                return full_result
            try:
                result = self.vlm_model.run(prompt, d2c_image, [])
                answer = result[0]
                print(answer)
                error = False
            except Exception as e:
                error_count += 1
                print('Error:', e)
                print('Retrying...')
        full_result[0].append(answer)
        full_result[1].append(result[1:])  # tokens info
        return full_result

    def mad_moderate_extractive(self, d2c_image, start_data=None):
        number = int(len(start_data[0]) / 2)
        first_reason = start_data[0][-2]
        second_reason = start_data[0][-1]
        prompt = self.prompt.format(number=number, first_reason=first_reason, second_reason=second_reason)
        full_result = start_data.copy()

        error = True
        error_count = 0
        while error:
            if error_count > 5:
                print("Too many errors, stopping execution.")
                full_result[0].append(f'{self.must_word}')
                full_result[1].append([])
                return full_result
            try:
                result = self.vlm_model.run(prompt, d2c_image, [])
                answer = result[0]
                print(answer)
                if self.must_word in answer:
                    error = False
                else:
                    error_count += 1
                    print("did not return the expected format. Retrying...")
            except Exception as e:
                error_count += 1
                print('Error:', e)
                print('Retrying...')
        full_result[0].append(answer)
        full_result[1].append(result[1:])  # tokens info
        return full_result