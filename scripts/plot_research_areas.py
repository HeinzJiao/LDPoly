import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyArrow

# 读取荷兰边界 GeoJSON 文件
# 下载地址：https://gadm.org/download_country.html (level-0：国家级；level-1：省级；level-2：市级)
netherlands = gpd.read_file('gadm41_NLD_0.json').to_crs(epsg=3857)

# 三个研究地点坐标（经纬度），转换成 GeoDataFrame
places = {
    "Deventer": (6.1552, 52.2550),
    "Enschede": (6.8958, 52.2215),
    "Giethoorn": (6.0886, 52.7426),
}
places_gdf = gpd.GeoDataFrame(
    {"name": list(places.keys())},
    geometry=[Point(xy) for xy in places.values()],
    crs="EPSG:4326"
).to_crs(epsg=3857)

# 开始画图
fig, ax = plt.subplots(figsize=(12, 14))

# 添加卫星底图之前，先画空白背景，避免白边
ax.set_facecolor('white')

# 绘制荷兰边界线
netherlands.plot(ax=ax, facecolor="none", edgecolor='white', linewidth=2, zorder=3)

# 绘制地标点
places_gdf.plot(ax=ax, color='red', markersize=80, zorder=4)

# 添加文字标注，白色字体+黑色描边
for idx, row in places_gdf.iterrows():
    txt = ax.text(
        row.geometry.x - 10000,  # 向左偏移一点
        row.geometry.y - 10000,
        row['name'],
        fontsize=18,
        weight='bold',
        color='white',
        ha='right',
        va='center',
        zorder=5
    )
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

# 加卫星底图
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zorder=0)

# 添加比例尺
scalebar = ScaleBar(1, units="m", location='upper right', scale_loc='bottom',
                    length_fraction=0.15, box_alpha=0, frameon=False, color='white')
ax.add_artist(scalebar)

# 添加指北针（左上角）
xlim = ax.get_xlim()
ylim = ax.get_ylim()
x_n = xlim[0] + 0.1 * (xlim[1] - xlim[0])
y_n = ylim[1] - 0.1 * (ylim[1] - ylim[0])
north_text = ax.text(x_n, y_n + 30000, 'N', fontsize=16, ha='center', color='white', weight='bold')
north_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
ax.arrow(x_n, y_n - 15000, 0, 20000, head_width=20000, head_length=20000,
         fc='white', ec='white', zorder=5)

# 设置显示范围为整个荷兰边界的范围
minx, miny, maxx, maxy = netherlands.total_bounds
ax.set_xlim(minx - 10000, maxx + 10000)
ax.set_ylim(miny - 40000, maxy + 10000)

# 去掉边框坐标轴
ax.axis('off')

# 保存高清图片
plt.savefig('research_areas_map_final.png', dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()
