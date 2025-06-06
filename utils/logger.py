import logging
import os 

def intialise_logger_nd_create_folders(args):
    # if args.model_path:
        # logging.info('resumed training')
        # num_folder=args.model_path.split(os.sep)[1]
        # print(num_folder)
        # save_path=args.save_dir+os.sep+num_folder
    # else:
    if os.path.exists(args.save_dir):
            num_folder=str(int(sorted(os.listdir(args.save_dir),key=lambda x:int(x))[-1])+1)
    else:
            os.makedirs(args.save_dir,exist_ok=False)
            num_folder='0'
    save_path=args.save_dir+os.sep+num_folder
    os.makedirs(save_path)


    os.makedirs(args.log_dir,exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,                  
        format='%(asctime)s - %(levelname)s - %(message)s',  
        datefmt='%Y-%m-%d %H:%M:%S',        
        handlers=[
            # logging.StreamHandler(),        
            logging.FileHandler(args.log_dir+os.sep+f"train_{num_folder}.log")  
        ]
    )
    print('logging at:',args.log_dir+os.sep+f"train_{num_folder}.log")
    logging.info('------------------------------------------------------------------------------------------------')
    logging.info('starting new training !!!!')
    logging.info('------------------------------------------------------------------------------------------------')

    logging.info(args)

        
    logging.info(f'parameters are being saved at :{save_path}')
    
    return save_path