/**
 * training_chart.js
 * -----------------
 * Reusable ApexCharts wrapper for real-time loss tracking.
 */

function initTrainingLossChart(containerSelector, initialData = []) {
  const seriesData = initialData.map(d => ({ x: d.step, y: d.loss }));

  const chartOptions = {
    chart: {
      type: 'area',
      height: 260,
      animations: { enabled: true, easing: 'linear', dynamicAnimation: { speed: 800 } },
      toolbar: { show: false },
      background: 'transparent'
    },
    series: [{ name: 'Training Loss', data: seriesData }],
    xaxis: {
      type: 'numeric',
      title: { text: 'Optimization Step' },
      labels: { formatter: val => parseInt(val) }
    },
    yaxis: {
      title: { text: 'Loss' },
      labels: { formatter: val => parseFloat(val).toFixed(3) }
    },
    stroke: { curve: 'smooth', width: 2 },
    colors: ['#0d6efd'],
    fill: {
      type: 'gradient',
      gradient: { shadeIntensity: 1, opacityFrom: 0.4, opacityTo: 0.05, stops: [0, 90, 100] }
    },
    dataLabels: { enabled: false },
    tooltip: {
      theme: 'dark',
      y: { formatter: val => val.toFixed(4) }
    }
  };

  const chart = new ApexCharts(document.querySelector(containerSelector), chartOptions);
  chart.render();

  return {
    chart,
    appendPoint: (step, loss) => {
      chart.appendData([{ data: [{ x: step, y: loss }] }]);
    },
    updateSeries: (data) => {
      chart.updateSeries([{ data: data.map(d => ({ x: d.step, y: d.loss })) }]);
    }
  };
}

window.initTrainingLossChart = initTrainingLossChart;
