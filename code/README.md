This is the code for [Self-supervised learning of class embeddings from video](https://arxiv.org/abs/1910.12699) in ICCV workshop in 2019.

Note that this code has not been 'cleaned' and so is only given for further explanation of the paper.

It's run by calling

`python train_attention_hierarchy.py --use_cyclic --use_confidence`.`

**Training yourself**

In order to use this training code, it is necessary to download a dataset (e.g. [VoxCeleb1/2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)).
They should then be put into folders as follows and the environment variables in Datasets/config.sh updated appropriately (VOX_CELEB_1 is VoxCeleb1, VOX_CELEB_LOCATION VoxCeleb2).

For our datasets we organised the directories as:

```
IDENTITY
-- VIDEO
-- -- TRACK
-- -- -- frame0001.jpg
-- -- -- frame0002.jpg
-- -- -- ...
-- -- -- frameXXXX.jpg
```


If you arrange the folders/files as illustrated above, then you can generate np split files using `Datasets/generate_large_voxceleb.py` and use our dataloader.
Otherwise, you may have to write your own.

Then you need to update where the model/runs are stored to by setting BASE_LOCATION.
