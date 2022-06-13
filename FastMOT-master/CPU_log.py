import psutil  
import GPUtil 

def write_log(CPU):
    file_name = 'CPU_log.txt'
    with open(file_name,'a+') as file:
        file.writelines(str(duration))

    return 0


def write_log(GPU):
    file_name = 'CPU_log.txt'
    with open(file_name,'a+') as file:
        file.writelines(str(duration))

    return 0
    

cpu_usage = psutil.cpu_percent()