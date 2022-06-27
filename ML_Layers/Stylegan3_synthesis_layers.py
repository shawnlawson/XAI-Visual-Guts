# Stylegan 3 Synthesis Layers

['L0_36_512', 
'L1_36_512',
'L2_52_512',
'L3_52_512',
'L4_84_512',
'L5_148_512',
'L6_148_512',
'L7_276_323',
'L8_276_203',
'L9_532_128',
'L10_1044_81',
'L11_1044_51',
'L12_1044_32',
'L13_1024_32',
'L14_1024_3']

SynthesisNetwork(
  w_dim=512,  num_ws=16,
  img_resolution=1024, img_channels=3,
  num_layers=14, num_critical=2,
  margin_size=10, num_fp16_res=4
  (input): SynthesisInput(
    w_dim=512, channels=512, size=[36, 36],
    sampling_rate=16, bandwidth=2
    (affine): FullyConnectedLayer(in_features=512, out_features=4, activation=linear)
  )
  (L0_36_512): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=False,
    in_sampling_rate=16, out_sampling_rate=16,
    in_cutoff=2, out_cutoff=2,
    in_half_width=6, out_half_width=6,
    in_size=[36, 36], out_size=[36, 36],
    in_channels=512, out_channels=512
    (affine): FullyConnectedLayer(in_features=512, out_features=512, activation=linear)
  )
  (L1_36_512): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=False,
    in_sampling_rate=16, out_sampling_rate=16,
    in_cutoff=2, out_cutoff=3.1748,
    in_half_width=6, out_half_width=4.8252,
    in_size=[36, 36], out_size=[36, 36],
    in_channels=512, out_channels=512
    (affine): FullyConnectedLayer(in_features=512, out_features=512, activation=linear)
  )
  (L2_52_512): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=False,
    in_sampling_rate=16, out_sampling_rate=32,
    in_cutoff=3.1748, out_cutoff=5.03968,
    in_half_width=4.8252, out_half_width=10.9603,
    in_size=[36, 36], out_size=[52, 52],
    in_channels=512, out_channels=512
    (affine): FullyConnectedLayer(in_features=512, out_features=512, activation=linear)
  )
  (L3_52_512): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=False,
    in_sampling_rate=32, out_sampling_rate=32,
    in_cutoff=5.03968, out_cutoff=8,
    in_half_width=10.9603, out_half_width=8,
    in_size=[52, 52], out_size=[52, 52],
    in_channels=512, out_channels=512
    (affine): FullyConnectedLayer(in_features=512, out_features=512, activation=linear)
  )
  (L4_84_512): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=False,
    in_sampling_rate=32, out_sampling_rate=64,
    in_cutoff=8, out_cutoff=12.6992,
    in_half_width=8, out_half_width=19.3008,
    in_size=[52, 52], out_size=[84, 84],
    in_channels=512, out_channels=512
    (affine): FullyConnectedLayer(in_features=512, out_features=512, activation=linear)
  )
  (L5_148_512): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=True,
    in_sampling_rate=64, out_sampling_rate=128,
    in_cutoff=12.6992, out_cutoff=20.1587,
    in_half_width=19.3008, out_half_width=43.8413,
    in_size=[84, 84], out_size=[148, 148],
    in_channels=512, out_channels=512
    (affine): FullyConnectedLayer(in_features=512, out_features=512, activation=linear)
  )
  (L6_148_512): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=True,
    in_sampling_rate=128, out_sampling_rate=128,
    in_cutoff=20.1587, out_cutoff=32,
    in_half_width=43.8413, out_half_width=32,
    in_size=[148, 148], out_size=[148, 148],
    in_channels=512, out_channels=512
    (affine): FullyConnectedLayer(in_features=512, out_features=512, activation=linear)
  )
  (L7_276_323): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=True,
    in_sampling_rate=128, out_sampling_rate=256,
    in_cutoff=32, out_cutoff=50.7968,
    in_half_width=32, out_half_width=77.2032,
    in_size=[148, 148], out_size=[276, 276],
    in_channels=512, out_channels=323
    (affine): FullyConnectedLayer(in_features=512, out_features=512, activation=linear)
  )
  (L8_276_203): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=True,
    in_sampling_rate=256, out_sampling_rate=256,
    in_cutoff=50.7968, out_cutoff=80.6349,
    in_half_width=77.2032, out_half_width=47.3651,
    in_size=[276, 276], out_size=[276, 276],
    in_channels=323, out_channels=203
    (affine): FullyConnectedLayer(in_features=512, out_features=323, activation=linear)
  )
  (L9_532_128): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=True,
    in_sampling_rate=256, out_sampling_rate=512,
    in_cutoff=80.6349, out_cutoff=128,
    in_half_width=47.3651, out_half_width=128,
    in_size=[276, 276], out_size=[532, 532],
    in_channels=203, out_channels=128
    (affine): FullyConnectedLayer(in_features=512, out_features=203, activation=linear)
  )
  (L10_1044_81): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=True,
    in_sampling_rate=512, out_sampling_rate=1024,
    in_cutoff=128, out_cutoff=203.187,
    in_half_width=128, out_half_width=308.813,
    in_size=[532, 532], out_size=[1044, 1044],
    in_channels=128, out_channels=81
    (affine): FullyConnectedLayer(in_features=512, out_features=128, activation=linear)
  )
  (L11_1044_51): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=False, use_fp16=True,
    in_sampling_rate=1024, out_sampling_rate=1024,
    in_cutoff=203.187, out_cutoff=322.54,
    in_half_width=308.813, out_half_width=189.46,
    in_size=[1044, 1044], out_size=[1044, 1044],
    in_channels=81, out_channels=51
    (affine): FullyConnectedLayer(in_features=512, out_features=81, activation=linear)
  )
  (L12_1044_32): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=True, use_fp16=True,
    in_sampling_rate=1024, out_sampling_rate=1024,
    in_cutoff=322.54, out_cutoff=512,
    in_half_width=189.46, out_half_width=118.346,
    in_size=[1044, 1044], out_size=[1044, 1044],
    in_channels=51, out_channels=32
    (affine): FullyConnectedLayer(in_features=512, out_features=51, activation=linear)
  )
  (L13_1024_32): SynthesisLayer(
    w_dim=512, is_torgb=False,
    is_critically_sampled=True, use_fp16=True,
    in_sampling_rate=1024, out_sampling_rate=1024,
    in_cutoff=512, out_cutoff=512,
    in_half_width=118.346, out_half_width=118.346,
    in_size=[1044, 1044], out_size=[1024, 1024],
    in_channels=32, out_channels=32
    (affine): FullyConnectedLayer(in_features=512, out_features=32, activation=linear)
  )
  (L14_1024_3): SynthesisLayer(
    w_dim=512, is_torgb=True,
    is_critically_sampled=True, use_fp16=True,
    in_sampling_rate=1024, out_sampling_rate=1024,
    in_cutoff=512, out_cutoff=512,
    in_half_width=118.346, out_half_width=118.346,
    in_size=[1024, 1024], out_size=[1024, 1024],
    in_channels=32, out_channels=3
    (affine): FullyConnectedLayer(in_features=512, out_features=32, activation=linear)
  )
)
