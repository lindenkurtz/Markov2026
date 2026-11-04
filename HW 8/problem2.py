import numpy as np

rng = np.random.default_rng(5)

def sim1(t, R):
    star_emitted = rng.poisson(3 * t, R)
    sky_emitted = rng.poisson(2 * t, R)

    star_kept = rng.binomial(star_emitted, 0.4)
    sky_kept = rng.binomial(sky_emitted, 0.4)
    return star_kept, sky_kept

def sim2(t, R):
    star_kept = rng.poisson(3 * t * 0.4, R)
    sky_kept = rng.poisson(2 * t * 0.4, R)
    return star_kept, sky_kept

star_1, sky_1 = sim1(5, 2 * 10 ** 5)
star_2, sky_2 = sim2(5, 2 * 10 ** 5)

print("sim1:")
print(f'means) star: {star_1.mean()} sky: {sky_1.mean()}')
print(f'var)   star: {star_1.var()} sky: {sky_1.var()}\n')
print("sim2:")
print(f'means) star: {star_2.mean()} sky: {sky_2.mean()}')
print(f'var)   star: {star_2.var()} sky: {sky_2.var()}\n')

corr1 = np.corrcoef(star_1, sky_1)[0, 1]
print(f'corr: {corr1}')

poiss_10 = rng.poisson(10, 2 * 10 ** 5)
total_detections = (star_1 + sky_1)
print(f'Poisson(10) mean/var: {poiss_10.mean()}, {poiss_10.var()}  total sim1 detections mean/var: {total_detections.mean()}, {total_detections.var()}')

detect_10 = (total_detections == 10)
exactly_6_star = (star_1[detect_10] == 6)
print(f'P(exactly 6 star | 10 total detections): {exactly_6_star.mean()}')

first_detect_time_star = rng.exponential(1 / 1.2, 2 * 10 ** 5)
first_detect_time_sky = rng.exponential(1 / 0.8, 2 * 10 ** 5)

star_first = (first_detect_time_star < first_detect_time_sky)
print(f'P(star first): {star_first.mean()}')
first_detect_time = np.minimum(first_detect_time_star, first_detect_time_sky)
print(f'P(star first and within 1 s): {(star_first & (first_detect_time < 1)).mean()}')
M = np.maximum(first_detect_time_star, first_detect_time_sky)
print(f'E[M]: {M.mean()}')
corr2 = np.corrcoef(first_detect_time, star_first.astype(float))[0, 1]
print(f'corr between time of first detection and type: {corr2}')