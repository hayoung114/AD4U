// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Pie Chart Example
var ctx = document.getElementById("myPieChart");
var myPieChart = new Chart(ctx, {
  type: 'pie',
  data: {
    labels: kinds,
    datasets: [{
      data: kinds_cnt,
      backgroundColor: ['#ffc107', '#ffc107c7', '#ffc1078f', '#ffc1074f', '#ffc10724'],
    }],
  },
});