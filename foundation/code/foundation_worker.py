"""Model-process CLI. Never import ManiSkill into the model's Python environment."""
import argparse,importlib,os
from multiprocessing.connection import Listener
from foundation_common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','train','serve']);parser.add_argument('job')
    args=parser.parse_args();job=read(args.job)
    if args.mode=='audit':
        data=RGBData(job['data'],job['cfg']['seed']);write(job['output'],data.audit());data.close();print('RGB dataset audit saved');return
    module=importlib.import_module('foundation_'+job['cfg']['backend'])
    if args.mode=='train':module.train(job);return
    policy=module.Policy(job['cfg'],job['checkpoint'])
    auth=bytes.fromhex(os.environ['MOVEBOXES_IPC_KEY'])
    with Listener(('127.0.0.1',0),authkey=auth) as listener:
        write(job['ready'],dict(host='127.0.0.1',port=listener.address[1]))
        with listener.accept() as connection:
            while True:
                request=connection.recv()
                if request['mode']=='close':break
                if request['mode']=='reset':policy.reset(request['seed']);connection.send({'ok':True});continue
                if request['mode']!='act':raise ValueError('Unknown policy request')
                action=policy.act(unwire_array(request['image']),unwire_array(request['state']))
                connection.send(dict(action=np.asarray(action,dtype=np.float32).tolist()))


if __name__=='__main__':main()
