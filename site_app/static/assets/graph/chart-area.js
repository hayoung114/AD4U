//{% load static %}
// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Area Chart Example

var ctx = document.getElementById("myAreaChart");
var myLineChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: ["0-2시", "2-4시", "4-6시", "6-8시", "8-10시", "10-12시", "12-14시", "14-16시", "16-18시", "18-20시", "20-22시", "22-24시"],
    datasets: [{
      label: "유동인구수",
      lineTension: 0.3,
      backgroundColor: "rgba(162, 82, 2, 0.32)",
      borderColor: "rgba(163, 103, 46, 1)",
      pointRadius: 5,
      pointBackgroundColor: "rgba(163, 103, 46, 1)",
      pointBorderColor: "rgba(255,255,255,0.8)",
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "(163, 103, 46, 1)",
      pointHitRadius: 50,
      pointBorderWidth: 2,
      data : cnt,
    }],
  },
  options: {
    scales: {
      xAxes: [{
        time: {
          unit: 'date'
        },
        gridLines: {
          display: false
        },
        ticks: {
          maxTicksLimit: 7
        }
      }],
      yAxes: [{
        ticks: {
          min: 0,
          max: 800,
          maxTicksLimit: 5
        },
        gridLines: {
          color: "rgba(0, 0, 0, .125)",
        }
      }],
    },
    legend: {
      display: false
    }
  }
});
