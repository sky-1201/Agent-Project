import os


import requests
from datetime import datetime
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

rag = RagSummarizeService()
external_data = {}

# 注意：在真实的系统中，USER_ID 通常不需要通过 Tool 让大模型去猜或随机生成，
# 而是直接从您之前重构的 FastAPI `request.user_id` 或者上下文中直接读取。
# 这里暂时保留您的结构，但在生产中建议直接传参。
user_ids = ["1001", "1002", "1003", "1004", "1005"]


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(
    description="获取指定城市的天气，包含【当前实时天气】以及【未来3天的天气预报】。当用户询问今天、明天、后天或未来天气时，请调用此工具。")
def get_weather(city: str) -> str:
    """
    更新版：适配 2026 年和风天气最新 API Host 机制
    """
    # 1. 填入你的真实 API KEY
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "57b4c85867e349598a9e8542065ced17")

    # 2. 【重点新增】填入你专属的 API Host (去控制台->设置里找，不要带 https://)
    API_HOST = os.getenv("WEATHER_API_HOST", "q43yftbwpt.re.qweatherapi.com")

    try:
        # 继续保持绕过系统代理，防止 VPN 干扰
        proxies = {"http": None, "https": None}

        # 1. 查城市 ID (使用你的专属 API_HOST 替换掉了旧的 geoapi)
        geo_url = f"https://{API_HOST}/v2/city/lookup?location={city}&key={WEATHER_API_KEY}"
        geo_response = requests.get(geo_url, timeout=5, proxies=proxies)
        geo_res = geo_response.json()

        if str(geo_res.get("code")) != "200":
            return f"未找到名为 {city} 的城市信息"

        location_id = geo_res["location"][0]["id"]

        # 2. 查询实时天气 (使用你的专属 API_HOST 替换掉了旧的 devapi)
        now_url = f"https://{API_HOST}/v7/weather/now?location={location_id}&key={WEATHER_API_KEY}"
        now_res = requests.get(now_url, timeout=5, proxies=proxies).json()

        # 3. 查询未来 3 天天气预报 (使用你的专属 API_HOST)
        forecast_url = f"https://{API_HOST}/v7/weather/3d?location={location_id}&key={WEATHER_API_KEY}"
        forecast_res = requests.get(forecast_url, timeout=5, proxies=proxies).json()

        # --- 拼装给大模型的综合天气信息 ---
        result_text = f"以下是【{city}】的天气信息：\n\n"

        if str(now_res.get("code")) == "200":
            now = now_res["now"]
            result_text += f"【当前实时】：{now['text']}，气温 {now['temp']}℃，体感 {now['feelsLike']}℃，相对湿度 {now['humidity']}%，{now['windDir']} {now['windScale']}级。\n"

        if str(forecast_res.get("code")) == "200":
            daily_forecasts = forecast_res["daily"]
            result_text += "【未来3天预报】：\n"
            for day in daily_forecasts:
                result_text += f"- {day['fxDate']}: 白天 {day['textDay']}，夜间 {day['textNight']}，气温 {day['tempMin']}℃ ~ {day['tempMax']}℃。\n"

        return result_text

    except Exception as e:
        logger.error(f"[get_weather] 请求天气API异常: {str(e)}")
        return f"查询{city}天气时发生异常，请稍后再试。"


@tool(description="获取用户当前所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """
    高德地图 IP 定位方案 (国内最准)
    """
    # 建议将 KEY 配置在环境变量或 agent.yml 中
    AMAP_KEY = os.getenv("AMAP_KEY", "7df35ae865eb97a4472cdd1cdcbbaf27")

    if AMAP_KEY == "7df35ae865eb97a4472cdd1cdcbbaf27":
        return "深圳"  # 如果没配Key，先给个兜底测试城市

    try:
        # 调用高德 IP 定位接口
        url = f"https://restapi.amap.com/v3/ip?key={AMAP_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()

        # status 为 1 表示请求成功
        if data.get("status") == "1":
            city = data.get("city", "")
            # 有时候直辖市的 city 可能是个空列表，需要取 province
            if not city or (isinstance(city, list) and len(city) == 0):
                city = data.get("province", "未知城市")

            logger.info(f"[get_user_location] 高德定位成功: {city}")
            return city

        logger.warning(f"[get_user_location] 高德定位失败: {data.get('info')}")
        return "北京"  # 接口报错时的兜底城市

    except Exception as e:
        logger.error(f"[get_user_location] 获取地理位置发生网络异常: {str(e)}")
        return "北京"


@tool(description="获取当前的绝对真实时间月份，以纯字符串形式返回，格式为YYYY-MM")
def get_current_month() -> str:
    """
    真实系统时间方案
    """
    # 直接通过 Python 内置时间库获取当前运行时的真实年月
    current_time = datetime.now()
    month_str = current_time.strftime("%Y-%m")
    return month_str


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    import random
    return random.choice(user_ids)


# ... 下方保留您原本的 generate_external_data 和 fetch_external_data, fill_context_for_report 的代码不变 ...




def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
