import pandas as pd
import plotly.express as px

# 数据定义
# data = {
#     "Category": ["Teeth", "Patho", "HisT", "Jaw", "SumRec", "Jaw", "Teeth", "HisT", "Patho", "Report", "SumRec"],
#     "Value": [ 359, 153, 145, 131, 32, 111, 321, 141, 115, 100, 65],
#     "Level": [ "Closed-Ended", "Closed-Ended", "Closed-Ended", "Closed-Ended", "Closed-Ended", 
#               "Open-ended", "Open-ended", "Open-ended", "Open-ended", "Open-ended", "Open-ended"],
# }

labels = ["MMOral", "Closed-Ended", "Open-Ended", "Teeth", "Patho", "HisT", "Jaw", "SumRec", "Jaw_", "Teeth_", "HisT_", "Patho_", "Report", "SumRec_"]
parents = ["", "MMOral", "MMOral", "Closed-Ended", "Closed-Ended", "Closed-Ended", "Closed-Ended", "Closed-Ended", "Open-Ended", "Open-Ended", "Open-Ended", "Open-Ended", "Open-Ended", "Open-Ended"]
values = [1673, 820, 853, 359, 153, 145, 131, 32, 111, 321, 141, 115, 100, 65]
special_font_sizes = {
        "MMOral": 30,
        "Teeth": 20,
        "Patho": 20,
        "HisT": 20,
        "Jaw": 20,
        "SumRec": 20,
        "Teeth_": 20,
        "Patho_": 20,
        "HisT_": 20,
        "Jaw_": 20,
        "SumRec_": 20,
        "Report": 20,
        "Open-Ended": 20,
        "Closed-Ended": 20,
    }
colors = ["white", "#00cc96", "#ffa15a",
           "#9299fc", "#f48876", "#4ddbb5", "#ffbd8b", "#c492fc", 

           "#ffbd8b", "#9299fc", "#4ddbb5", "#f48876",  "#98df8a", "#c492fc"]

fig = px.sunburst(
        names=labels,
        parents=parents,
        values=values,
        # title="Attribute Distribution",
        branchvalues='total',
        width=1000,
        height=1000,
    )


fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        # uniformtext=dict(minsize=12, mode='hide'),
        font_family="Times New Roman",  # 全局字体
    )

fig.update_traces(textinfo="label+percent entry",
                  sort=True,
                  selector=dict(type='sunburst'),
                textfont=dict(
                    family="Times New Roman",
                    size=[special_font_sizes.get(label, 30) for label in labels],
                    color="black"
                    ),
                    marker=dict(
                        colors=colors
                    )
                )


fig.write_image('MMOral-Bench_dist.svg', format='svg')

