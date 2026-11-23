## 噪声标定的流程，得到跟ISO相关的噪声标定参数K，B。

1. 有很多ISO档位，比如100, 200, 400, 800, 1600, 23000等，依次处理ISO，得到跟ISO相关的噪声标定参数K，B。
2. 选择一个ISO档位，有很多曝光时间，比如100us, 200us, 400us, 800us, 1600us, 2300us等，依次处理曝光时间，得到均值（X）和方差（Y），Y=KX+B。
3. 选择一个曝光时间，里面有亮帧跟黑帧，对亮帧跟黑帧进行处理，得到均值（X）和方差（Y）。
4. 亮帧有两张RAW图，先求均值，每张图把所有像素相加后除以 w * h，再把同组两张图的均值取平均，得到亮帧均值。
5. 两张亮帧逐像素相减得到差值图。两帧均值之差就是差值图的均值，用它把差值图去均值，再取平方的平均并除以 2，得到亮帧方差。
6. 黑帧跟亮帧一样操作得到黑帧均值和黑帧方差。
7. 亮帧均值减去黑帧均值，得到均值，作为X。
8. 亮帧方差作为Y。
9. 看（X，Y）的散点图，选择前25个点，后面11个点不要了，然后做拟合得到K，B。

## 拟合ISO跟K，一次多项式方程
1. 上面求了每个ISO的K，B，现在要拟合ISO跟K。
2. 直接用ISO作为x，K作为y，做一次多项式拟合，得到K=Cx+D。
3. C是拟合系数，D是拟合截距。
4. ISO跟B也可以拟合，是二次多项式拟合，目前没有做。

## 得到参数
1. 有很多ISO档位，比如100, 200, 400, 800, 1600, 23000等，依次处理ISO。
2. 选择一个ISO档位，就一个20ms的曝光时间，过滤存在越界码值的帧，假设有450张黑帧，则450 × 1536 × 1920。
3. 对450 × 1536 × 1920 转换为float32，然后减黑电平blc，得到450 × 1536 × 1920。
4. 对450 × 1536 × 1920 求平均，得到1536 x 1920，记为固定暗场偏差darkshading。
5. 对1536 × 1920按照RGGB拆成四个通道 768 × 960 × 4，记为raw。
6. 然后对raw求平均，得到(4,)，记为mean_raw。
7. 然后raw - mean_raw，从各通道的每个像素中减去该通道的均值，得到768 × 960 × 4，记为noise_map。

8. 坏点校正，是下面第9--12步，用中值滤波结果替换疑似坏点，或者直接选择第13步，保留原始图样，代码选择的是第十三步。
9. denoised = cv2.medianBlur(noise_map, 5)，分别在每个通道内做5 × 5中值滤波，得到768 × 960 × 4，中值结果用于比较某个像素是否明显偏离周围像素，记为denoised。
10. sigma_dark = (noise_map - denoised).std(axis=(0, 1))，计算原始图与中值图的差值，再分别求四个通道中这些差值的空间标准差，得到(4,)，这四个数用于设定各通道的坏点检测阈值，记为sigma_dark。
11. bad_points_map = abs(noise_map - denoised) > 6 * sigma_dark，逐像素比较差值的绝对值是否超过所属通道标准差的6倍，得到768 × 960 × 4的bool mask：True表示疑似坏点，记为bad_points_map。
12. dark_shading = denoised * bad_points_map + noise_map * (1-bad_points_map)，在bad_points_map为True的位置，用中值滤波结果替换原值；其他位置保留noise_map中的原值，得到的768 × 960 × 4，记为dark_shading。

13. dark_shading = noise_map，直接保留原始的通道去均值图样，得到768 × 960 × 4，记为dark_shading。

14. dark_shading加回mean_raw，得到768 × 960 × 4，记为dark_shading。
15. dark_shading还原成1536 × 1920的二维 Bayer 图，得到1536 × 1920，记为darkshading。要是走第十三步的话，就是第四步。

16. 遍历450帧1536 × 1920 raw图，以一个1536 × 1920 raw图为例.
17. 减黑电平，减darkshading，得到新的1536 × 1920 raw图。
18. 对1536 × 1920按照RGGB拆成四个通道 768 × 960 × 4，记为packed_raw。
19. 每个通道单独求平均，每个通道得到一个值，记为color_bias。
20. 裁剪图片到400 * 400 * 4 ，记为packed_raw_crop。

21. 下面第22--32步里面是单个通道，单个通道来处理的
22. _, (sigRead, uRead, rRead) = scipy.stats.probplot(img.reshape(-1), rvalue=True)，将 160000 个像素值拉成一维并排序，以标准正态分布的理论分位数为横坐标、排序后的实际像素值为纵坐标，做最小二乘直线拟合，返回斜率、截距和相关系数。
23. 将 `sigRead` 存到 `sigmaRead[c, i]`，表示去行均值之前的整体噪声尺度，包含行相关波动，`sigRead` 是拟合斜率，不是直接调用 `np.std()` 的结果；数据接近正态分布时，可作为标准差估计。
24. 将 `uRead` 存到 `meanRead[c, i]`，表示拟合中心位置。
25. 将 `rRead**2` 存到 `r2Read[c, i]`，表示概率图的直线拟合程度。
26. 对每行的 400 个像素求平均，得到形状为 (400,) 的行均值数组，记为row_all。
27. _, (sigR, uR, rR) = scipy.stats.probplot(row_all, rvalue=True)，对这 400 个行均值做正态概率图拟合。
28. 保存 `sigmaR[c, i] = sigR`、`meanR[c, i] = uR`、`r2R[c, i] = rR**2`。`sigR` 是行均值之间波动的尺度，用作行噪声强度估计，不是每一行内部像素的标准差；行内像素噪声平均后的残留也会影响它。
29. 400 * 400每行的全部像素减去该行自己的均值，形状仍为 400 × 400，去除前包含这一行的共同偏移，去除后只保留各像素相对本行均值的波动，每行均值约为零。
30. 400 * 400将去行均值后的图拉成长度为 160000 的一维数组，记为X。
31. lam = scipy.stats.ppcc_max(X)，搜索使实际数据与 Tukey Lambda 理论分位数相关性最高的形状参数 λ，保存为 `lamda[c, i]`。λ 决定分布形状，是无量纲参数，不是均值、方差或噪声强度。
32. _, (sigGs, uGs, rGs) = scipy.stats.probplot(X, rvalue=True)。
33. 保存 `sigmaGs[c, i] = sigGs`、`meanGs[c, i] = uGs`、`r2Gs[c, i] = rGs**2`。含义分别是去行均值后剩余噪声的高斯尺度估计、中心位置、拟合程度。它与第 33 步是对同一份 X 的两种分布描述，不是两份独立噪声。
33. _, (sigTL, uTL, rTL) = scipy.stats.probplot(X, dist=scipy.stats.tukeylambda(lam), rvalue=True)，用 Tukey Lambda 理论分位数与实际排序值做直线拟合。
34. 保存 `sigmaTL[c, i] = sigTL`、`meanTL[c, i] = uTL`、`r2TL[c, i] = rTL**2`。`sigTL` 是该分布的尺度 scale，`uTL` 是位置 loc，`rTL**2` 是拟合程度。`sigTL` 不能直接当作标准差；对应模型的方差为 `sigTL**2 * scipy.stats.tukeylambda.var(lam)`，前提是该 λ 下方差存在。

35. 当前帧的四个通道完成后，处理下一帧，重复第 16～34 步，直到450帧全部完成。
36. 各噪声尺度、中心位置、拟合程度、λ 和 `color_bias` 数组均为 4 × N，还没有跨帧汇总为一组数。
37. wp = WHITE_LEVEL、bl = BLACK_LEVEL。
38. 所有ISO都得到这些值后，开始拟合跨ISO的模板和噪声配置。

39. 依次处理每个ISO，从第39步到41步。
40. 对当前 ISO 的1536 × 1920的darkshading做整张图的平均，得到一个值。
41. 然后1536 × 1920的darkshading减去这个值，将每个 ISO 的结果依次放入darkshading1。
42. 每个 ISO 与对应的 darkshading1 一一对应，比如ISO有29个值，记为legal_iso。
43. darkshading1从 29 × 1536 × 1920 变成 29 × 2949120，记为ds_data。
44. reg = np.polyfit(legal_iso, ds_data, deg=1)，以 ISO 为横坐标，对全部 2949120 个像素位置分别拟合，得到形状为 2 × 2949120 的 reg。 `reg[0]`、`reg[1]` 分别还原成 1536 × 1920 的 `ds_k`、`ds_b`。这个版本是直接拟合。

45. 依次处理每个ISO，从第44步到50步。
46. 对1536 × 1920的darkshading拆成4个通道，768 × 960 × 4，记为ds。
47. 对每个通道单独做cv2.medianBlur(ds, 5)，5 * 5中值滤波。
48. 四个通道合并成1536 × 1920的darkshading2。
49. 对1536 × 1920的darkshading2做整张图的平均，得到一个值。
50. 然后1536 × 1920的darkshading2减去这个值，得到新的darkshading3。
51. 每个 ISO 与对应的 darkshading3 一一对应，比如ISO有29个值，记为legal_iso1。
52. darkshading3从 29 × 1536 × 1920 变成 29 × 2949120，记为ds_data1。
53. reg1 = np.polyfit(legal_iso1, ds_data1, deg=1)，以 ISO 为横坐标，对全部 2949120 个像素位置分别拟合，得到形状为 2 × 2949120 的 reg1。 `reg1[0]`、`reg1[1]` 分别还原成 1536 × 1920 的 `ds_k1`、`ds_b1`。这个版本是先滤波再拟合。

54. 依次处理每个ISO，从第55步到69步。
55. 对4 * 450 的color_bias做平均，得到4，记为color_bias1。
56. 对4 * 450 的sigmaRead做平均，得到450，记为sigmaRead1。
57. 对4 * 450 的meanRead做平均，得到450，然后求一次标准差，得到一个值uRead。
58. 对4 * 450 的r2Read做平均，得到450。
59. 对4 * 450 的sigmaR做平均，得到450，记为sigmaR1。
60. 对4 * 450 的meanR做平均，得到450。
61. 对4 * 450 的r2R做平均，得到450。
62. 对4 * 450 的sigmaGs做平均，得到450，记为sigmaGs1。
63. 对4 * 450 的meanGs做平均，得到450。
64. 对4 * 450 的r2Gs做平均，得到450。
65. 对4 * 450 的sigmaTL做平均，得到450，记为sigmaTL1。
66. 对4 * 450 的meanTL做平均，得到450。
67. 对4 * 450 的r2TL做平均，得到450。
68. 对4 * 450 的lam做平均，得到450，记为lam1，每个ISO复制450次，拟合 lam1=E*ISO+F。
69. 用已经标定并写入配置的关系计算 `K = ISO * K_ISO[0] + K_ISO[1]`，就是前面的 `K = C * ISO + D`。记为point_iso。

70. ISO按照升序排列，再把对应的color_bias1按ISO顺序排列，得到29 * 4，记为color_bias_iso。
71. 拿所有ISO的sigmaR1与K配对，sigmaR_all有29 * 450个值，K_all有29 * 450（每个ISO复制450次），sigmaR1 450个值都是同一个K，拟合regr.fit(np.log(K_all).reshape(-1, 1),np.log(sigmaR_all))。
72. 拿所有ISO的sigmaTL1与K配对，sigmaTL_all有29 * 450个值，K_all有29 * 450（每个ISO复制450次），sigmaTL1 450个值都是同一个K，拟合regr.fit(np.log(K_all).reshape(-1, 1),np.log(sigmaTL_all))。
73. 拿所有ISO的sigmaGs1与K配对，sigmaGs_all有29 * 450个值，K_all有29 * 450（每个ISO复制450次），sigmaGs1 450个值都是同一个K，拟合regr.fit(np.log(K_all).reshape(-1, 1),np.log(sigmaGs_all))。
74. 拿所有ISO的sigmaRead1与K配对，sigmaRead_all有29 * 450个值，K_all有29 * 450（每个ISO复制450次），sigmaRead1 450个值都是同一个K，拟合regr.fit(np.log(K_all).reshape(-1, 1),np.log(sigmaRead_all))。
75. 拿所有ISO的uRead拟合，uRead=G*K+H。



