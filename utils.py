import numpy as np
import matplotlib.pyplot as plt
import imageio
from copy import deepcopy
from tqdm import tqdm

# For the convenience of visualization, 
# we only consider optimization function of 2 D


def GDA_update_step(alpha, nabla_x, nabla_y):
    """
    构造一个function, 这个function输入 x, y, 生成 x_{t+1}, y_{t+1}
    
    x_{t+1} = x_t - \alpha \nabla_x f(x_t, y_t)
    y_{t+1} = y_t + \alpha \nabla_y f(x_t, y_t)
    
    x_t, y_t are float type
    alpha is the learning rate
    nabla_x, nabla_y are the function that takes in the x_t and y_t that returns float
    
    return the next step of x and y
    """
    def update(x_t, y_t):
        x_t_plus_1 = x_t - alpha * nabla_x(x_t, y_t)
        y_t_plus_1 = y_t + alpha * nabla_y(x_t, y_t)
        return x_t_plus_1, y_t_plus_1
    return update


def OGDA_update_step(alpha, nabla_x, nabla_y):
    """
    构造一个function, 这个function输入 x, y, 生成 x_{t+1}, y_{t+1}
    
    x_{t+1} = x_t - 2*\alpha \nabla_x f(x_t, y_t) + \alpha \nabla_x f(x_{t-1}, y_{t-1})
    y_{t+1} = y_t + 2*\alpha \nabla_y f(x_t, y_t) - \alpha \nabla_y f(x_{t-1}, y_{t-1})
    
    x_t, y_t are float type
    alpha is the learning rate
    nabla_x, nabla_y are the function that takes in the x_t and y_t that returns float
    
    return the next step of x and y
    
    
    Special Attention: For the first step, we use the GDA update rule
    Thus we only need to let x_t_minus_1 = x_t, y_t_minus_1 = y_t
    """
    def update(x_t, y_t, x_t_minus_1, y_t_minus_1):
        x_t_plus_1 = x_t - 2 * alpha * nabla_x(x_t, y_t) + alpha * nabla_x(x_t_minus_1, y_t_minus_1)
        y_t_plus_1 = y_t + 2 * alpha * nabla_y(x_t, y_t) - alpha * nabla_y(x_t_minus_1, y_t_minus_1)
        return x_t_plus_1, y_t_plus_1
    return update


def optimize(x_0_lis, y_0_lis, update_function, steps, func_folder_path, kernel, bound=20):
    """
    This function takes in the initial x and y for many initializations, the update function, the number of steps and the folder path to save the optimization data
    x_0, y_0: float
    update_function: GDA_update_step or OGDA_update_step
    steps: int, the number of steps
    func_folder_path: str, the folder path to save the optimization data
    """
    x_current_lis, y_current_list = x_0_lis, y_0_lis
    # x_lis[i][j] is the x of the j-th state at the i-th step
    x_lis = [x_current_lis.copy()]
    y_lis = [y_current_list.copy()]
    import os
    os.makedirs(f'{func_folder_path}/{kernel}', 
                exist_ok=True)

    if kernel == "GDA":
        print('GDA updating---')
        for _ in tqdm(range(steps)):
            for i in range(len(x_current_lis)):
                x_current_lis[i], y_current_list[i] = update_function(x_current_lis[i], y_current_list[i])
                if x_current_lis[i] > bound:
                    x_current_lis[i] = np.nan
                if y_current_list[i] > bound:
                    y_current_list[i] = np.nan
            x_lis.append(x_current_lis.copy())
            y_lis.append(y_current_list.copy())
    elif kernel == "OGDA":
        """
        OGDA 第一步的更新公式使用 GDA 的更新公式
        OGDA 的更新公式为
        x_{t+1} = x_t - \alpha \nabla_x f(x_t, y_t)
        y_{t+1} = y_t + \alpha \nabla_y f(x_t, y_t)
        """
        pass

		# 第一步先用 GDA 的更新公式
        for i in range(len(x_current_lis)):
            x_current_lis[i], y_current_list[i] = update_function(x_t = x_current_lis[i], 
                                                                  y_t = y_current_list[i],
                                                                  x_t_minus_1 = x_current_lis[i],
                                                                  y_t_minus_1 = y_current_list[i])
            if x_current_lis[i] > bound:
                x_current_lis[i] = np.nan
            if y_current_list[i] > bound:
                y_current_list[i] = np.nan
        x_lis.append(x_current_lis.copy())
        y_lis.append(y_current_list.copy())
        print('OGDA_updating---')
        for _ in tqdm(range(steps - 1)):
            for i in range(len(x_current_lis)):
                x_current_lis[i], y_current_list[i] = update_function(x_t = x_current_lis[i], 
																	  y_t = y_current_list[i],
                                                                      x_t_minus_1 = deepcopy( x_lis[-2][i]),
                                                                      y_t_minus_1 = deepcopy( y_lis[-2][i]))
																	#   x_t_minus_1 = x_lis[-2][i].copy(),
																	#   y_t_minus_1 = y_lis[-2][i].copy())
                if x_current_lis[i] > bound:
                    x_current_lis[i] = np.nan
                if y_current_list[i] > bound:
                    y_current_list[i] = np.nan
            x_lis.append(x_current_lis.copy())
            y_lis.append(y_current_list.copy())
    
    x_data = np.array(x_lis)
    y_data = np.array(y_lis)
    np.save(f'{func_folder_path}/{kernel}/x_data.npy', x_data)
    np.save(f'{func_folder_path}/{kernel}/y_data.npy', y_data)
    return


def get_timestep_images(func_folder_path, kernel, img_step=5, func_exp=None):
    """
    This function uses the optimization data to generate the images of the optimization process
    func_folder_path: str, the folder path to save the optimization data
    kernel: str, the kernel name, kernel in ["GDA", "OGDA"]
    img_step: int, the step of the images, AKA, there's no need for us to create a image for every step
    we create image per img_step steps
    """
    x_lis = np.load(f'{func_folder_path}/{kernel}/x_data.npy')
    y_lis = np.load(f'{func_folder_path}/{kernel}/y_data.npy')
    x_min, x_max = np.nanmin(x_lis), np.nanmax(x_lis)
    y_min, y_max = np.nanmin(y_lis), np.nanmax(y_lis)
    _state_number = len(x_lis[0])
    _opt_steps = x_lis.shape[0]

    import os
    os.makedirs(f'{func_folder_path}/{kernel}/images', exist_ok=True)
    print('image generation---')
    for i in tqdm(range(0, _opt_steps, img_step)):
        import os
        if os.path.exists(f'{func_folder_path}/{kernel}/images/step_{i}.png'):
            continue
        plt.figure(figsize=(10, 10))
        for j in range(_state_number):
            plt.plot(x_lis[i][j], y_lis[i][j], 'ro')
        
        # TODO(BobHunagC): there no need to draw all the trajectory, for normal step, just keep for 100 steps
        for _state_idx in range(len(x_lis[0])):
            # draw the trajectory of the state
            # if it isn't the last opt_step, then draw the last 100 steps of trajectories
            if i != _opt_steps - 1:
                _tmp_draw_x_lis = [x_lis[_time_idx][_state_idx] for _time_idx in range(max(0, i-100), i+1)]
                _tmp_draw_y_lis = [y_lis[_time_idx][_state_idx] for _time_idx in range(max(0, i-100), i+1)]
            # if it is the last opt_step, then draw the full trajectory
            elif i == _opt_steps - 1:
                _tmp_draw_x_lis = [x_lis[_time_idx][_state_idx] for _time_idx in range(0, i+1)]
                _tmp_draw_y_lis = [y_lis[_time_idx][_state_idx] for _time_idx in range(0, i+1)]			
            plt.plot(_tmp_draw_x_lis, _tmp_draw_y_lis, 'bo-', markersize=0.1)

        # plt.xlim(max(x_min - 0.2, -0.5), min(x_max + 0.2, 1.5))
        plt.xlim(max(x_min - 0.2, -2), min(x_max + 0.2, 2))
        # plt.xlim(-20, 20)
        # plt.ylim(max(y_min - 0.2, -0.5), min(y_max + 0.2, 1.5))
        plt.ylim(max(y_min - 0.2, -2), min(y_max + 0.2, 2))
        # plt.ylim(-20, 20)
        if func_exp:
            plt.title(f'Step {i} | {func_exp}')
        else:
            plt.title(f'Step {i}')
        plt.savefig(f'{func_folder_path}/{kernel}/images/step_{i}.png')
        plt.close()


def gif_generation(func_folder_path, 
                   kernel,
                   fps = 10, 
                   gif_img_step = 5):
    """
    This function use the images in the path of func_folder_path/kernel/images to create a GIF
	func_folder_path: str, the folder path to save data related to the function
    kernel: str, the kernel name, kernel in ["GDA", "OGDA"]
    fps:
    gif_img_step: int, the step of choosing imgs to consist gif, there's no need to create a gif for every step
    """
    images = []

    import os
    def get_all_file_paths(directory):
        import re
        def get_number_from_file_path(file_path):
            # 使用正则表达式找到文件名末尾的数字
            match = re.search(r'(\d+)(?=\.\w+$)', file_path)
            # 如果找到了数字，返回这个数字
            # 否则，返回0
            return int(match.group()) if match else 0
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return sorted(file_paths,  key=get_number_from_file_path)
    
    images_paths = get_all_file_paths(f'{func_folder_path}/{kernel}/images')
    print('gif generation-----')
    for i in tqdm(range(0, len(images_paths), gif_img_step)):
        # images.append(imageio.imread(f'{images_folder_path}/step_{i}.png'))
        images.append(imageio.imread(images_paths[i]))
    # imageio.mimsave(f'{images_folder_path}-optimization.gif', images, fps=fps)
    imageio.mimsave(f'{func_folder_path}/{kernel}/{kernel}-optimization.gif', images, fps=fps)