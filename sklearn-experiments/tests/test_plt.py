from utils.plt_utils import plot, plot_scatter, plot_bar


class TestMatplotlib():
    def test_plot(self):
        x = [1, 2, 3, 4]
        y = [2, 4, 6, 8]
        plot(x, y, 'test_plot')

    def test_scatter(self):
        x = [1, 2, 3, 4]
        y = [2, 4, 6, 8]
        plot_scatter(x, y, 'test_scatter')

    def test_bar(self):
        x = [1, 2, 3, 4]
        y = [2, 4, 6, 8]
        plot_bar(x, y, 'test_bar')


if __name__ == '__main__':
    ins = TestMatplotlib()
    ins.test_plot()
    ins.test_scatter()
    ins.test_bar()