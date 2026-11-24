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

## clean raw图加噪声
主流程采用“光子散粒噪声 + 行噪声 + 像素噪声 + 帧间整体偏移”；像素噪声在高斯和 Tukey Lambda 中选择一种。

76. 读取一张clean RAW，尺寸1536 × 1920。
77. 按照ISO选择darkshading，然后减黑电平，减去darkshading，最小值限制到0。得到新的clean_signal。
78. 按照ISO选择K，记为K1。
79. 对1536 × 1920做RGGB拆分通道，得到4个通道，768 × 960 × 4。记为clean_packed。
80. 由第71步的行噪声尺度回归，计算 `sigmaR_syn = np.exp(sigRk * np.log(K1) + sigRb)`。其中 `sigRk` 是第71步回归得到的斜率，`sigRb` 是截距。
81. 如果选择高斯像素噪声，由第73步的回归计算 `sigmaGs_syn = np.exp(sigGsk * np.log(K1) + sigGsb)`。它用于描述去掉行均值后剩余像素噪声的标准差估计。
82. 如果选择Tukey Lambda像素噪声，由第72步的回归计算 `sigmaTL_syn = np.exp(sigTLk * np.log(K) + sigTLb)`。它是Tukey Lambda的scale，不是标准差，不再对它开平方。
83. Tukey Lambda还需要形状参数 `lam_syn`。虽然第68步拟合了λ与ISO的关系，但当前全ISO参数记录保存的是全部逐帧λ的平均值，即 `lam_syn = 全ISO的lam1拼接后的均值`，对应导出字段 `lam`。本流程使用这个均值，不使用 `E * ISO + F`，也不把λ写死为某个数。
84. 由第75步计算 `uRead_syn = G * K + H = uReadk * K + uReadb`，其中 `uReadk = G`、`uReadb = H`。这里的uRead_syn是帧中心值波动的标准差估计，不是要直接加到图上的固定偏移；作为抽样尺度时必须非负。
85. 建立随机数生成器 `rng = np.random.default_rng(seed)`。逐像素计算 `electrons_mean = clean_packed / K`，再执行 `shot_packed = K * rng.poisson(electrons_mean)`，得到包含干净信号和光子散粒噪声的图。若想单独查看噪声，则 `N_shot = shot_packed - clean_packed`；在未裁剪的线性域中，这部分的期望为0，逐像素方差为 `K * clean_packed`。
86. 使用第80步的sigmaR_syn抽取行噪声：`N_row = rng.normal(loc=0, scale=sigmaR_syn, size=(H//2, 1, 4))`，本例为768 × 1 × 4。每个子通道每行只抽一个值，沿列方向广播到整行，而不是为行内每个像素重新抽样。四个子通道共享前面平均后得到的尺度，但各自抽取随机行偏移。
87. 高斯像素噪声方案：`N_pixel = rng.normal(loc=0, scale=sigmaGs_syn, size=clean_packed.shape)`。每个像素抽一个值，均值设为0，因为用于标定sigmaGs的数据已经减过各行均值。不要把meanGs再作为一份额外噪声叠加。
88. Tukey Lambda像素噪声方案：`N_pixel = scipy.stats.tukeylambda.rvs(lam=lam_syn, loc=0, scale=sigmaTL_syn, size=clean_packed.shape, random_state=rng)`。第87、88步只执行一个，两者都是对同一份去行均值残差的建模。Tukey Lambda的方差为 `sigmaTL_syn**2 * scipy.stats.tukeylambda.var(lam_syn)`，前提是对应方差存在。
89. 使用第84步的uRead_syn抽取帧间整体偏移：`N_frame = rng.normal(loc=0, scale=uRead_syn)`，得到一个标量。把这个标量加到整张图、四个子通道的所有像素上。同一帧只抽一次，下一帧重新抽一次；不能按像素抽，否则就变成另一份像素噪声了。当前uRead是四通道中心值平均后的波动，因此这里只建模四通道共享的整体偏移。
90. 合成 `noisy_packed = shot_packed + N_row + N_pixel + N_frame`，形状仍为H/2 × W/2 × 4。等价写法是 `clean_packed + N_shot + N_row + N_pixel + N_frame`。shot_packed已经包含clean_packed，不能再额外加一次clean_packed。
91. `noisy_corrected = rggb2bayer(noisy_packed)`，还原成H × W的二维Bayer图。它与clean_signal一样处于已扣黑电平、已扣固定暗场偏差的信号域，可以直接作为这一数值约定下的加噪结果。保存浮点结果时保留负噪声值，不要为了显示而先裁剪到0。
92. 如果目标是还原未经校正的传感器RAW，需要加回目标ISO的偏置。原始模板方案为 `DS_target = ds_k * ISO + ds_b + BLE[ISO]`：ds_k、ds_b来自第44步，BLE[ISO]是第40步减掉并保存的整图均值。若使用第53步的滤波模板ds_k1、ds_b1，就配套使用第49步保存的滤波后BLE。两套模板选择一套，不能相加；本例先选择已标定档位，保证对应BLE可以查到。
93. 未校正的RAW为 `noisy_raw_float = noisy_corrected + bl + DS_target`，其中bl是第37步的 `BLACK_LEVEL`，DS_target本身不包含这个bl。如果后续模型输入本来就已经扣过这些偏置，可以使用第91步结果，不需要先加回再重复扣除。
94. 输出当前12bit整数RAW时，最后执行 `noisy_raw = np.clip(np.rint(noisy_raw_float), 0, wp).astype(np.uint16)`，其中wp是第37步的 `WHITE_LEVEL`，当前为4095。这一步是DN域的取整和饱和处理。本流程不再额外叠加一份均匀量化噪声；导出的 `q = 1 / wp` 是归一化尺度下的量化步长约定，不能直接把这个数当成DN域噪声幅度来加。
95. 前面的第80～84步先使用拟合曲线给出的典型尺度，不必使用所有残差参数。如果希望增加同一ISO下的尺度变化，可以额外采用高斯残差近似，例如每个合成帧先抽一次 `log_sigmaR_syn = rng.normal(sigRk * np.log(K) + sigRb, sigRsig)`，然后 `sigmaR_syn = np.exp(log_sigmaR_syn)`；sigmaGs、sigmaTL同理使用sigGssig、sigTLsig。uReadsig属于未取对数的uRead回归残差，只能用于其尺度的线性域扰动，并保证扰动后的标准差非负。这里的残差扰动是额外的合成假设，不是现有脚本已经验证或实现的生成逻辑。
96. 当前采用行噪声、像素噪声分别生成的方案，因此不再用第74步的sigmaRead模型额外加一份噪声，也不再用最开始拟合的B额外加一份噪声。meanRead已经用于得到uRead，meanR、meanGs、meanTL不再逐项相加。color_bias_iso是模板扣除后的通道均值残差，本主流程不额外叠加它；恢复固定的通道均值偏差时已使用配套的darkshading与BLE。各r2参数只用于检查拟合，不参与抽样。
97. 保存clean/noisy配对数据，并记录目标ISO、K、seed、选择的像素分布、实际使用的sigmaR_syn、sigmaGs_syn或sigmaTL_syn、lam_syn、uRead_syn，以及是否包含bl和DS_target。clean和noisy的训练输入必须采用一致的数值域、偏置与归一化约定；有意保留固定偏差作为去除目标时，则应单独写明这种输入/目标差异。RAW数组直接保存，不经过白平衡、去马赛克或Gamma转换。
98. 同一张clean可以换seed生成多张noise RAW。每一张重新抽取N_shot、N_row、N_pixel、N_frame；同一ISO下的固定暗场模板DS_target保持不变。使用真实黑帧对比像素残差分布、行均值波动、帧均值波动，再检查不同亮度下的均值—方差关系，确认合成结果与标定相符。统计对比应在同一校正域、相同子通道和裁剪口径下进行，并避开已经饱和或被裁剪的样本。
