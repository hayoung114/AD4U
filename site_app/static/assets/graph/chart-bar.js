//{% load static %}
// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
var ctx = document.getElementById("myBarChart");
var myLineChart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ["1-7여자", "1-7남자", "8-13여자", "8-13남자", "14-19여자", "14-19남자", "20-25여자", "20-25남자", "26-33여자", "26-33남자", "34-50여자", "34-50남자", "51-70여자", "51-70남자", "70-100여자", "70-100남자"],
    datasets: [{
      label: "유동인구수",
      backgroundColor: "rgba(102, 191, 34, 0.87)",
      borderColor: "rgba(102, 191, 34, 0.87)",
      data: target_cnt,
    }],
  },
  options: {
    scales: {
      xAxes: [{
        time: {
          unit: 'target'
        },
        gridLines: {
          display: false
        },
        ticks: {
          maxTicksLimit: 18
        }
      }],
      yAxes: [{
        ticks: {
          min: 0,
          max: 300,
          maxTicksLimit: 5
        },
        gridLines: {
          display: true
        }
      }],
    },
    legend: {
      display: false
    }
  }
});
