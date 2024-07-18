// 
import React, { memo, useEffect, useRef, useState } from 'react';

import * as echarts from 'echarts';
import { EChartsOption } from 'echarts';
import {
  // 直角坐标系内绘图网格组件
  TitleComponentOption, // 提示框组件
  TooltipComponentOption,
} from 'echarts/components';

echarts.registerTheme('shine', {
  "color": [
      "#c12e34",
      "#e6b600",
      "#0098d9",
      "#2b821d",
      "#005eaa",
      "#339ca8",
      "#cda819",
      "#32a487"
  ],
  "backgroundColor": "rgba(0,0,0,0)",
  "textStyle": {},
  "title": {
      "textStyle": {
          "color": "#333333"
      },
      "subtextStyle": {
          "color": "#aaaaaa"
      }
  },
  "line": {
      "itemStyle": {
          "borderWidth": 1
      },
      "lineStyle": {
          "width": 2
      },
      "symbolSize": 4,
      "symbol": "emptyCircle",
      "smooth": false
  },
  "radar": {
      "itemStyle": {
          "borderWidth": 1
      },
      "lineStyle": {
          "width": 2
      },
      "symbolSize": 4,
      "symbol": "emptyCircle",
      "smooth": false
  },
  "bar": {
      "itemStyle": {
          "barBorderWidth": 0,
          "barBorderColor": "#ccc"
      }
  },
  "pie": {
      "itemStyle": {
          "borderWidth": 0,
          "borderColor": "#ccc"
      }
  },
  "scatter": {
      "itemStyle": {
          "borderWidth": 0,
          "borderColor": "#ccc"
      }
  },
  "boxplot": {
      "itemStyle": {
          "borderWidth": 0,
          "borderColor": "#ccc"
      }
  },
  "parallel": {
      "itemStyle": {
          "borderWidth": 0,
          "borderColor": "#ccc"
      }
  },
  "sankey": {
      "itemStyle": {
          "borderWidth": 0,
          "borderColor": "#ccc"
      }
  },
  "funnel": {
      "itemStyle": {
          "borderWidth": 0,
          "borderColor": "#ccc"
      }
  },
  "gauge": {
      "itemStyle": {
          "borderWidth": 0,
          "borderColor": "#ccc"
      }
  },
  "candlestick": {
      "itemStyle": {
          "color": "#c12e34",
          "color0": "#2b821d",
          "borderColor": "#c12e34",
          "borderColor0": "#2b821d",
          "borderWidth": 1
      }
  },
  "graph": {
      "itemStyle": {
          "borderWidth": 0,
          "borderColor": "#ccc"
      },
      "lineStyle": {
          "width": 1,
          "color": "#aaaaaa"
      },
      "symbolSize": 4,
      "symbol": "emptyCircle",
      "smooth": false,
      "color": [
          "#c12e34",
          "#e6b600",
          "#0098d9",
          "#2b821d",
          "#005eaa",
          "#339ca8",
          "#cda819",
          "#32a487"
      ],
      "label": {
          "color": "#eeeeee"
      }
  },
  "map": {
      "itemStyle": {
          "areaColor": "#ddd",
          "borderColor": "#eee",
          "borderWidth": 0.5
      },
      "label": {
          "color": "#c12e34"
      },
      "emphasis": {
          "itemStyle": {
              "areaColor": "#e6b600",
              "borderColor": "#ddd",
              "borderWidth": 1
          },
          "label": {
              "color": "#c12e34"
          }
      }
  },
  "geo": {
      "itemStyle": {
          "areaColor": "#ddd",
          "borderColor": "#eee",
          "borderWidth": 0.5
      },
      "label": {
          "color": "#c12e34"
      },
      "emphasis": {
          "itemStyle": {
              "areaColor": "#e6b600",
              "borderColor": "#ddd",
              "borderWidth": 1
          },
          "label": {
              "color": "#c12e34"
          }
      }
  },
  "categoryAxis": {
      "axisLine": {
          "show": true,
          "lineStyle": {
              "color": "#333"
          }
      },
      "axisTick": {
          "show": true,
          "lineStyle": {
              "color": "#333"
          }
      },
      "axisLabel": {
          "show": true,
          "color": "#333"
      },
      "splitLine": {
          "show": false,
          "lineStyle": {
              "color": [
                  "#ccc"
              ]
          }
      },
      "splitArea": {
          "show": false,
          "areaStyle": {
              "color": [
                  "rgba(250,250,250,0.3)",
                  "rgba(200,200,200,0.3)"
              ]
          }
      }
  },
  "valueAxis": {
      "axisLine": {
          "show": true,
          "lineStyle": {
              "color": "#333"
          }
      },
      "axisTick": {
          "show": true,
          "lineStyle": {
              "color": "#333"
          }
      },
      "axisLabel": {
          "show": true,
          "color": "#333"
      },
      "splitLine": {
          "show": true,
          "lineStyle": {
              "color": [
                  "#ccc"
              ]
          }
      },
      "splitArea": {
          "show": false,
          "areaStyle": {
              "color": [
                  "rgba(250,250,250,0.3)",
                  "rgba(200,200,200,0.3)"
              ]
          }
      }
  },
  "logAxis": {
      "axisLine": {
          "show": true,
          "lineStyle": {
              "color": "#333"
          }
      },
      "axisTick": {
          "show": true,
          "lineStyle": {
              "color": "#333"
          }
      },
      "axisLabel": {
          "show": true,
          "color": "#333"
      },
      "splitLine": {
          "show": true,
          "lineStyle": {
              "color": [
                  "#ccc"
              ]
          }
      },
      "splitArea": {
          "show": false,
          "areaStyle": {
              "color": [
                  "rgba(250,250,250,0.3)",
                  "rgba(200,200,200,0.3)"
              ]
          }
      }
  },
  "timeAxis": {
      "axisLine": {
          "show": true,
          "lineStyle": {
              "color": "#333"
          }
      },
      "axisTick": {
          "show": true,
          "lineStyle": {
              "color": "#333"
          }
      },
      "axisLabel": {
          "show": true,
          "color": "#333"
      },
      "splitLine": {
          "show": true,
          "lineStyle": {
              "color": [
                  "#ccc"
              ]
          }
      },
      "splitArea": {
          "show": false,
          "areaStyle": {
              "color": [
                  "rgba(250,250,250,0.3)",
                  "rgba(200,200,200,0.3)"
              ]
          }
      }
  },
  "toolbox": {
      "iconStyle": {
          "borderColor": "#06467c"
      },
      "emphasis": {
          "iconStyle": {
              "borderColor": "#4187c2"
          }
      }
  },
  "legend": {
      "textStyle": {
          "color": "#333333"
      }
  },
  "tooltip": {
      "axisPointer": {
          "lineStyle": {
              "color": "#cccccc",
              "width": 1
          },
          "crossStyle": {
              "color": "#cccccc",
              "width": 1
          }
      }
  },
  "timeline": {
      "lineStyle": {
          "color": "#005eaa",
          "width": 1
      },
      "itemStyle": {
          "color": "#005eaa",
          "borderWidth": 1
      },
      "controlStyle": {
          "color": "#005eaa",
          "borderColor": "#005eaa",
          "borderWidth": 0.5
      },
      "checkpointStyle": {
          "color": "#005eaa",
          "borderColor": "#316bc2"
      },
      "label": {
          "color": "#005eaa"
      },
      "emphasis": {
          "itemStyle": {
              "color": "#005eaa"
          },
          "controlStyle": {
              "color": "#005eaa",
              "borderColor": "#005eaa",
              "borderWidth": 0.5
          },
          "label": {
              "color": "#005eaa"
          }
      }
  },
  "visualMap": {
      "color": [
          "#1790cf",
          "#a2d4e6"
      ]
  },
  "dataZoom": {
      "backgroundColor": "rgba(47,69,84,0)",
      "dataBackgroundColor": "rgba(47,69,84,0.3)",
      "fillerColor": "rgba(167,183,204,0.4)",
      "handleColor": "#a7b7cc",
      "handleSize": "100%",
      "textStyle": {
          "color": "#333333"
      }
  },
  "markPoint": {
      "label": {
          "color": "#eeeeee"
      },
      "emphasis": {
          "label": {
              "color": "#eeeeee"
          }
      }
  }
});

interface EChartsChartProps {
  content: string;
}

const EChartsChart: React.FC<EChartsChartProps> = memo(({ content }) => {
  const chartRef = useRef<HTMLDivElement>(null);

  const [chartJson, setChartJson] = useState<any>({});

  useEffect(() => {
    //console.log(content);
    const data = JSON.parse(content);
    setChartJson(data);
  }, []);

  useEffect(() => {
    let chartInstance: echarts.ECharts | null = null;
    console.log(chartJson);
    if (chartRef.current) {
      chartInstance = echarts.init(chartRef.current, 'shine');
      if ('color' in chartJson) delete chartJson['color'];
      if ('series' in chartJson) {
        for (let i = 0; i < chartJson['series'].length; ++i) {
          if ('lineStyle' in chartJson['series'][i]) {
            delete chartJson['series'][i]['lineStyle'];
          }
        }
      }
      if(chartJson.series && chartJson.series[0] && chartJson.series[0].type === 'pie'){
        chartJson['series'][0]['center'][0]="65%";
        chartJson['series'][0]['center'][1]="50%";
        chartJson['legend'][0]['left']= 'left';
        chartJson['legend'][0]['orient']= 'vertical';
      }
      chartInstance.setOption(chartJson as EChartsOption);
    }
    window.addEventListener('resize', handleResize);
    return () => {
      if (chartInstance) {
        chartInstance.dispose();
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [chartRef.current]);

  const handleResize = () => {
    if (chartRef.current) {
      const chartInstance = echarts.getInstanceByDom(chartRef.current);
      if (chartInstance) {
        chartInstance.resize();
      }
    }
  };

  return <div ref={chartRef} className="w-full h-[500px]" />;
});

export default EChartsChart;