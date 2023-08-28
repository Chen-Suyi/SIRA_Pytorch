import numpy as np
import nori2 as nori
import os
import shutil
import pickle
from tqdm import tqdm
from collections import defaultdict
from nori2 import NoriWriter


class NoriMaker:

    def __init__(self, out_path: str):
        self.out_path = out_path
        self.dict = defaultdict(list)

    def dump_nid_dict(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_nid_dict(self, path):
        with open(path, 'rb') as f:
            nid_dict = pickle.load(f)
        return nid_dict

    def make_nori(self, nw: NoriWriter, data: list, flag: str):

        for idx, sample in enumerate(tqdm(data)):

            if flag == 'modelnet40':
                points_bytes = sample['points'].tobytes()
                nid = nw.put(points_bytes)
                data = {'nid': nid, 'label': sample['label']}
                self.dict[sample['label']].append(data)

            elif flag == '7scenes_real':
                # cur_data = {'points_src': src_pc, 'points_ref': ref_pc, 'transform_gt': gt, 'category': sub_dir,
                #             'src_idx': src_idx, 'ref_idx': ref_idx}
                points_src_bytes = sample['points_src'].tobytes()
                points_src_nid = nw.put(points_src_bytes)
                points_ref_bytes = sample['points_ref'].tobytes()
                points_ref_nid = nw.put(points_ref_bytes)
                transform_gt_bytes = sample['transform_gt'].tobytes()
                transform_gt_nid = nw.put(transform_gt_bytes)
                data = {
                    'points_src_nid': points_src_nid,
                    'points_ref_nid': points_ref_nid,
                    'transform_gt_nid': transform_gt_nid,
                    'src_idx': sample['src_idx'],
                    'ref_idx': sample['ref_idx'],
                    'category': sample['category']
                }
                self.dict[sample['category']].append(data)

            elif flag == '7scenes_sync':
                points_bytes = sample['points'].tobytes()
                points_nid = nw.put(points_bytes)
                data = {
                    'points_nid': points_nid,
                    'idx': sample['idx'],
                    'category': sample['category']
                }
                self.dict[sample['category']].append(data)

            elif flag == 'kitti_real':
                # cur_data = {'points_src': src_pc, 'points_ref': ref_pc, 'transform_gt': gt, 'category': sub_dir,
                #             'src_idx': src_idx, 'ref_idx': ref_idx}
                points_src_bytes = sample['points_src'].tobytes()
                points_src_nid = nw.put(points_src_bytes)
                points_ref_bytes = sample['points_ref'].tobytes()
                points_ref_nid = nw.put(points_ref_bytes)
                pose_gt_bytes = sample['pose_gt'].tobytes()
                pose_gt_nid = nw.put(pose_gt_bytes)
                data = {
                    'points_src_nid': points_src_nid,
                    'points_ref_nid': points_ref_nid,
                    'pose_gt_nid': pose_gt_nid,
                    'src_idx': sample['src_idx'],
                    'ref_idx': sample['ref_idx'],
                    'category': sample['category']
                }
                self.dict[sample['category']].append(data)

            elif flag == 'kitti_sync':
                points_bytes = sample['points'].tobytes()
                points_nid = nw.put(points_bytes)
                data = {
                    'points_nid': points_nid,
                    'idx': sample['idx'],
                    'category': sample['category']
                }
                self.dict[sample['category']].append(data)
            else:
                raise NotImplementedError

    def make(self, split: str, data: list):
        if not os.path.exists(os.path.join(self.out_path, "{}".format(split))):
            os.mkdir(os.path.join(self.out_path, "{}".format(split)))
        else:
            print("Warning: dir {} already exists, let's remove it".format(
                split))
            shutil.rmtree(os.path.join(self.out_path, "{}".format(split)))
            os.mkdir(os.path.join(self.out_path, "{}".format(split)))

        with nori.open(
                os.path.join(self.out_path, "{}/data.nori".format(split)),
                "w") as nw:
            self.make_nori(nw, data)

        self.dump_nid_dict(
            os.path.join(self.out_path, '{}_files.pickle'.format(split)))

    def remote_make(self, split: str, data: list, flag: str):
        if not os.path.exists(os.path.join(self.out_path, "{}".format(split))):
            os.mkdir(os.path.join(self.out_path, "{}".format(split)))
        else:
            print("Warning: dir {} already exists, let's remove it".format(
                split))
            shutil.rmtree(os.path.join(self.out_path, "{}".format(split)))
            os.mkdir(os.path.join(self.out_path, "{}".format(split)))

        print('nori path on oss is : s3://xhdata' + '{}'.format(self.out_path))
        with nori.remotewriteopen(
                's3://xhdata' +
                '{}/{}/data.nori'.format(self.out_path, split)) as nw:
            self.make_nori(nw, data, flag)

        self.dump_nid_dict(
            os.path.join(self.out_path, '{}_nid.pickle'.format(split)))

    def to_oss_and_speedup(self, split):
        nori_path = os.path.join(self.out_path, "{}/data.nori".format(split))
        print('nori path: {}'.format(nori_path))

        target_path = 's3://xhdata' + nori_path
        print(target_path + ' speed up!')
        os.system('nori speedup ' + target_path + ' --on' + ' --replica 2')
