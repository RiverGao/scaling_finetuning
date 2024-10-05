import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from scipy import stats


model_type = 'bert'
model_size = 'large'
d_size_layer = {'gpt2-base': 12, 'gpt2-large': 36, 'gpt2-multi': 24, 'bert-base': 12, 'bert-large': 24}
n_layers = d_size_layer[model_type + '-' + model_size]
methods = ['pearson', 'spearman']

# table: {p1, p2} * {base, large}
# sheet: pearson, spearman
# row: layer
# columns: t_value(r_num), p_value(r_num), t_value(r_dur), p_value(r_dur)
with pd.ExcelWriter(f't-test/correlation/{model_type}/{model_size}/amr_{model_size}_ttest.xlsx') as writer:
    for method in methods:
        count_ttest = 0  # used for significance correction
        df_out = pd.DataFrame(columns=['layer', 't(r AMR)', 'p(r AMR)'])
        for layer in range(n_layers):
            df_model = pd.read_excel(
                f'correlation/{model_type}/{model_size}/amr_{model_size}_{method}.xlsx',
                sheet_name=f'layer {layer}')
            df_random = pd.read_excel(
                f'correlation/{model_type}/{model_size}-random/amr_{model_size}-random_{method}.xlsx',
                sheet_name=f'layer {layer}')

            r_amr_model = df_model['r AMR'].to_numpy()
            sign_amr_model = 1 if r_amr_model.mean() >= 0 else -1
            r_amr_random = df_random['r AMR'].to_numpy()
            sign_amr_random = 1 if r_amr_random.mean() >= 0 else -1

            t_amr, p_amr = stats.ttest_rel(sign_amr_model * r_amr_model,
                                           sign_amr_random * r_amr_random,
                                           alternative='greater')
            count_ttest += 1
            # print(t_amr, p_num)

            df_out = df_out.append(
                {'layer': layer, 't(r AMR)': t_amr, 'p(r AMR)': p_amr},
                ignore_index=True)

        df_out.to_excel(writer, sheet_name=method, index=False)

        significant_p = 0.05 / count_ttest
        print(f'Significant p value is {significant_p}')
