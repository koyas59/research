#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iterator>
#include <map>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

const int IMAGE_WIDTH = 640;
const int IMAGE_HEIGHT = 480;

class Utilities
{
public:
	/**
	* @brief �א�������
	*/
	static void Thinning(Mat &img)
	{
		const int width = img.cols;
		const int height = img.rows;

		cv::rectangle(img, cv::Point(0, 0), cv::Point(width - 1, height - 1), CV_RGB(0, 0, 0));


		// 4�ߖT�p�k�ޏ����p�̃J�[�l���𐶐�
		cv::Mat four_neighbor = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

		int step = 0; // �J��Ԃ���
		while (1) {
			step++;             // �J��Ԃ�����1���₷

			// ����f = 4�ߖT��0��f����ȏ㑶�݂���1��f ��I�яo��

			cv::Mat candidate = img.clone();
			cv::erode(candidate, candidate, four_neighbor);
			candidate = img - candidate;

			// ���ォ�珇���������āA���邩�A���Ȃ����𔻒f����
			int num_delete = 0;         // �������f�̐���ێ�
			for (int y = 1; y < height - 1; y++) {
				for (int x = 1; x < width - 1; x++) {
					if (candidate.at<uchar>(y, x) == 0) { // ���������?
						continue;     // �����łȂ���Ύ��̉�f�̏�����
					}
					// ���Ӊ�f�̏�Ԃ𒲂ׂ�
					int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
					int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
					unsigned char n[8];     // �אډ�f�̏��
					int num = 0;                    // �אډ�f��
					for (int i = 0; i < 8; i++) {
						n[i] = img.at<uchar>(y + dy[i], x + dx[i]) ? 1 : 0;
						num += n[i];
					}
					// 8�ߖT��"1"���O�ȏ�Ȃ�������A�����ł��Ȃ��̂Ŏ��̉�f��
					if (num < 3) {
						continue;
					}
					// �A�����𒲂ׂ�
					int m = 0;      // �A����
					for (int i = 0; i < 4; i++) {
						int k0 = i * 2;
						int k1 = (k0 + 1) % 8;
						int k2 = (k1 + 1) % 8;
						m += n[k0] * (1 - n[k1] * n[k2]);
					}
					// �A������1�Ȃ炻�̉�f�͏����\�Ȃ̂ŁA��������
					if (m == 1) {
						img.at<uchar>(y, x) = 0;
						num_delete++;
					}
				}
			}

			// �I������
			if (num_delete == 0) {              // ���f�����Ȃ�������
				break;                                    // �I��
			}
		}
	}

	/**
	* @brief 4�_poly���ɁC�_p�����݂����true
	*/
	static bool cn(const vector<Point> &poly, const Point p) {
		Point p1, p2;
		bool inside = false;
		Point oldPoint = poly[poly.size() - 1];

		for (int i = 0; i < poly.size(); i++) {
			Point newPoint = poly[i];
			if (newPoint.x > oldPoint.x) { p1 = oldPoint; p2 = newPoint; }
			else { p1 = newPoint; p2 = oldPoint; }
			if ((p1.x < p.x) == (p.x <= p2.x) && (p.y - p1.y)*(p2.x - p1.x) < (p2.y - p1.y)*(p.x - p1.x)) {
				inside = !inside;
			}
			oldPoint = newPoint;
		}
		return inside;
	}

	static void meanThreshold(const Mat src, Mat &dst, const vector<Point> &poly, vector< pair<int, int> > sRect, int kernelSize, double k, double r)
	{
		dst.create(src.size(), src.type());

		Mat srcWithBorder;
		int borderSize = kernelSize / 2;
		int kernelPixels = kernelSize * kernelSize;
		copyMakeBorder(src, srcWithBorder, borderSize, borderSize,
			borderSize, borderSize, cv::BORDER_REPLICATE);

		Mat sum, sqSum;
		integral(srcWithBorder, sum, sqSum);

		int x = 0, y = 0;

		cout << to_string(src.rows) + ',' + to_string(src.cols) + '\n';

		for (y = 0; y < src.rows; y++) {
			for (x = 0; x < src.cols; x++) {

				if (cn(poly, Point(x, y))) {
					int kx = x + kernelSize;
					int ky = y + kernelSize;
					double sumVal = sum.at<int>(ky, kx) - sum.at<int>(ky, x) - sum.at<int>(y, kx) + sum.at<int>(y, x);
					double sqSumVal = sqSum.at<double>(ky, kx) - sqSum.at<double>(ky, x) - sqSum.at<double>(y, kx) + sqSum.at<double>(y, x);

					double mean = sumVal / kernelPixels;
					double var = (sqSumVal / kernelPixels) - (mean * mean);
					if (var < 0.0) var = 0.0;
					double stddev = sqrt(var);
					double threshold = mean;

					if (src.at<uchar>(y, x) < threshold) {
						dst.at<uchar>(y, x) = 0;
					}
					else {
						dst.at<uchar>(y, x) = 255;
					}
				}
				else {
					dst.at<uchar>(y, x) = 128;
				}

			}
		}
	}

	static void niblack(const Mat src, Mat &dst, const vector<Point> &poly, vector< pair<int, int> > sRect, int kernelSize, double k, double r)
	{
		dst.create(src.size(), src.type());

		Mat srcWithBorder;
		int borderSize = kernelSize / 2;
		int kernelPixels = kernelSize * kernelSize;
		copyMakeBorder(src, srcWithBorder, borderSize, borderSize,
			borderSize, borderSize, cv::BORDER_REPLICATE);

		Mat sum, sqSum;
		integral(srcWithBorder, sum, sqSum);

		int x = 0, y = 0;

		cout << to_string(src.rows) + ',' + to_string(src.cols) + '\n';

		for (y = 0; y < src.rows; y++) {
			for (x = 0; x < src.cols; x++) {

				if (cn(poly, Point(x, y))) {
					int kx = x + kernelSize;
					int ky = y + kernelSize;
					double sumVal = sum.at<int>(ky, kx) - sum.at<int>(ky, x) - sum.at<int>(y, kx) + sum.at<int>(y, x);
					double sqSumVal = sqSum.at<double>(ky, kx) - sqSum.at<double>(ky, x) - sqSum.at<double>(y, kx) + sqSum.at<double>(y, x);

					double mean = sumVal / kernelPixels;
					double var = (sqSumVal / kernelPixels) - (mean * mean);
					if (var < 0.0) var = 0.0;
					double stddev = sqrt(var);
					double threshold = mean + k * stddev;

					if (src.at<uchar>(y, x) < threshold) {
						dst.at<uchar>(y, x) = 0;
					}
					else {
						dst.at<uchar>(y, x) = 255;
					}
				}
				else {
					dst.at<uchar>(y, x) = 128;
				}

			}
		}
	}

	static void sauvolaFast(const Mat src, Mat &dst, const vector<Point> &poly, int kernelSize, double k, double r)
	{
		dst.create(src.size(), src.type());
		dst = 60;

		Mat srcWithBorder;
		int borderSize = kernelSize / 2;
		int kernelPixels = kernelSize * kernelSize;
		copyMakeBorder(src, srcWithBorder, borderSize, borderSize,
			borderSize, borderSize, cv::BORDER_REPLICATE);

		Mat sum, sqSum;
		integral(srcWithBorder, sum, sqSum);

		int x = 0, y = 0;

		cout << to_string(src.rows) + ',' + to_string(src.cols) + '\n';

		for (y = 0; y < src.rows; y++) {
			for (x = 0; x < src.cols; x++) {

				if (cn(poly, Point(x, y))) {
					int kx = x + kernelSize;
					int ky = y + kernelSize;
					double sumVal = sum.at<int>(ky, kx) - sum.at<int>(ky, x) - sum.at<int>(y, kx) + sum.at<int>(y, x);
					double sqSumVal = sqSum.at<double>(ky, kx) - sqSum.at<double>(ky, x) - sqSum.at<double>(y, kx) + sqSum.at<double>(y, x);

					double mean = sumVal / kernelPixels;
					double var = (sqSumVal / kernelPixels) - (mean * mean);
					if (var < 0.0) var = 0.0;
					double stddev = sqrt(var);
					double threshold = mean * (1 + k * (stddev / r - 1));

					if (src.at<uchar>(y, x) < threshold) {
						dst.at<uchar>(y, x) = 0;
					}
					else {
						dst.at<uchar>(y, x) = 255;
					}
				}

			}
		}
	}

	static int MaxNumber(vector<int> vec)
	{
		if (vec.size() == 0) return 0;
		return *max_element(vec.begin(), vec.end());
	}

private:
	static void __drawHistgram__(Mat &histgram, vector<int> row_sum, vector<int> col_sum, pair<Point, Point> circumscribed_point)
	{
		for (int x = circumscribed_point.first.x; x < circumscribed_point.second.x; x++) {
			line(histgram, Point(x, 0), Point(x, row_sum[x]), Scalar(0, 0, 255), 1, 4);
		}

		for (int y = circumscribed_point.first.y; y < circumscribed_point.second.y; y++) {
			line(histgram, Point(0, y), Point(col_sum[y], y), Scalar(0, 0, 255), 1, 4);
		}
	}
};

class GuideBoard
{
public:

	static Mat GetHue(const Mat &src)
	{
		Mat dst;
		__RgbToHsv__(src, dst);

		vector<Mat> channel;
		split(dst, channel);
		return channel[0].clone();
	}

	static Mat GetSaturation(const Mat &src)
	{
		Mat dst;
		__RgbToHsv__(src, dst);

		vector<Mat> channel;
		split(dst, channel);
		return channel[1].clone();
	}

	static Mat GetValue(const Mat &src)
	{
		Mat dst;
		__RgbToHsv__(src, dst);

		vector<Mat> channel;
		split(dst, channel);
		return channel[2].clone();
	}

	/**
	* @brief �G�b�W�摜�̒��̍ʓx��threshold�ȉ��̃G�b�W�_����菜��
	*/
	static void RemoveEdgeOfLowSat(const Mat edgeImg, const Mat saturationImg, Mat &dst, int threshold, vector<Point> &removePoint)
	{
		dst = edgeImg.clone();
		removePoint.clear();

		Mat saturationImgWithBorder;
		copyMakeBorder(saturationImg, saturationImgWithBorder, 1, 1, 1, 1, cv::BORDER_REPLICATE);

		int x, y, satx, saty;
		for (y = 0; y < edgeImg.rows; y++) {
			for (x = 0; x < edgeImg.cols; x++) {
				satx = x + 1;
				saty = y + 1;

				if (
					(saturationImgWithBorder.at<uchar>(saty, satx) < threshold) &&
					(saturationImgWithBorder.at<uchar>(saty - 1, satx - 1) < threshold) &&
					(saturationImgWithBorder.at<uchar>(saty - 1, satx) < threshold) &&
					(saturationImgWithBorder.at<uchar>(saty - 1, satx + 1) < threshold) &&
					(saturationImgWithBorder.at<uchar>(saty, satx + 1) < threshold) &&
					(saturationImgWithBorder.at<uchar>(saty + 1, satx + 1) < threshold) &&
					(saturationImgWithBorder.at<uchar>(saty + 1, satx) < threshold) &&
					(saturationImgWithBorder.at<uchar>(saty + 1, satx - 1) < threshold) &&
					(saturationImgWithBorder.at<uchar>(saty, satx - 1) < threshold)
					) {
					dst.at<uchar>(y, x) = 0;
					removePoint.push_back(Point(x, y));
				}
			}
		}
	}

	/**
	* @brief �G�b�W�摜����C�l�p�`�̈��T���Ă���4���_�ƁA����4�_���܂ދ�`�̈��4���_���o��
	* @note  apexes�F4�_�łł���̈�̓_���W�̃x�N�g���CsRect�F4�_�łł���̈���͂ދ�`�̈�̓_���W�̃x�N�g��
	*/
	static vector< vector< Point > > FindApexesByEdge(const Mat edgeImg, const int areaThreshold)
	{
		vector<vector<Point>> apexes;
		vector<vector<Point>> contours;
		findContours(edgeImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		int j = 0;
		auto it = contours.begin();

		while (it != contours.end()) {
			if (contourArea(contours.at(distance(contours.begin(), it))) < areaThreshold) {
				it = contours.erase(it);
			}
			else {
				Mat contour = Mat(*it);

				vector<Point> approx;
				convexHull(contour, approx);

				// �֊s�E�|�����C���ߎ�
				approxPolyDP(approx, approx, 0.02 * cv::arcLength(contour, true), true);

				if (approx.size() == 4) { // ���_��4�̗֊s�݂̂��Ώ�
					apexes.push_back(approx); // �̈�����4�_�̍��W���i�[
				}

				j++;
				it++;
			}
		}

		return apexes;
	}

	static pair<Point, Point> CircumscribedPointOf(vector<Point> apex, int offset)
	{
		pair<int, int> x_minmax, y_minmax;

		x_minmax = minmax({ apex[0].x, apex[1].x, apex[2].x, apex[3].x });
		y_minmax = minmax({ apex[0].y, apex[1].y, apex[2].y, apex[3].y });

		pair<Point, Point> circumscribed_point = (
			make_pair(
			Point(
			x_minmax.first + offset, y_minmax.first + offset
			),
			Point(
			x_minmax.second - offset, y_minmax.second - offset
			)
			)
			);

		return circumscribed_point;
	}

	static vector< pair<Point, Point> > GetObjectArea(const Mat binarized_image, vector<Point> apex, int offset)
	{
		vector<int> x_histgram, y_histgram;
		pair<Point, Point> circumscribed_point = GuideBoard::CircumscribedPointOf(apex, offset);

		int min_x = circumscribed_point.first.x,
			max_x = circumscribed_point.second.x,
			min_y = circumscribed_point.first.y,
			max_y = circumscribed_point.second.y;

		Mat histgram = Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, Scalar(255, 255, 255));
		vector<int> row_sum(IMAGE_WIDTH, 0), col_sum(IMAGE_HEIGHT, 0);

		for (int y = min_y; y < max_y; y++) {
			for (int x = min_x; x < max_x; x++) {
				if (binarized_image.at<uchar>(y, x) == 0) {
					row_sum[x]++;
					col_sum[y]++;
				}
			}
		}

		__AdjustHistgram__(row_sum, 0.1, 5);
		__AdjustHistgram__(col_sum, 0.1, 5);

		__drawHistgram__(histgram, row_sum, col_sum, circumscribed_point);

		imwrite("histgram.png", histgram);

		vector< pair<Point, Point> > a;

		return a;
	}

	static vector< pair<Point, Point> > GetObjectArea2(const Mat binarized_image, vector<Point> apex, int offset)
	{
		vector<int> x_histgram, y_histgram;
		pair<Point, Point> circumscribed_point = GuideBoard::CircumscribedPointOf(apex, offset);

		int min_x = circumscribed_point.first.x,
			max_x = circumscribed_point.second.x,
			min_y = circumscribed_point.first.y,
			max_y = circumscribed_point.second.y;

		Mat histgram = Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, Scalar(255, 255, 255));
		Mat histgram_line = Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, Scalar(255, 255, 255));
		Mat histgram_area = Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, Scalar(255, 255, 255));
		Mat histgram_overlapped_area = Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, Scalar(255, 255, 255));

		/* �q�X�g�O�������v�Z */
		vector<int> row_sum(IMAGE_WIDTH, 0), col_sum(IMAGE_HEIGHT, 0);
		__ComputeHistgram__(binarized_image, min_x, max_x, min_y, max_y, row_sum, col_sum);

		/* �q�X�g�O�����𒲐� */
		//__AdjustHistgram__(row_sum, 0.1, 5);
		//__AdjustHistgram__(col_sum, 0.1, 5);

		/* �q�X�g�O�����̋�Ԃ������� */
		vector<pair<int, int>> interval_x, interval_y;
		interval_x = __ComputeInterval__(row_sum);
		interval_y = __ComputeInterval__(col_sum);

		/* �摜����� */
		__drawHistgram__(histgram, row_sum, col_sum, circumscribed_point);
		__drawInterval__(histgram_line, interval_x, interval_y);
		__drawAreas__(histgram_area, interval_x, interval_y);
		__drawOverlappedAreas__(histgram_overlapped_area, interval_x, interval_y);

		imwrite("hist.png", histgram);
		imwrite("hist_line.png", histgram_line);
		imwrite("hist_area.png", histgram_area);
		imwrite("hist_overlapped_area.png", histgram_overlapped_area);

		vector< pair<Point, Point> > a;

		return a;
	}


private:
	static void __RgbToHsv__(const Mat &src, Mat &dst)
	{
		Mat hsv_image;
		cvtColor(src, hsv_image, CV_BGR2HSV);
		dst = hsv_image.clone();
	}

	static void __AdjustHistgram__(vector<int> &histgram, double threshold_ratio, int threshold_series_num)
	{
		vector<int> series, series_num;

		for (int hi = 0; hi < histgram.size(); hi++) {
			if (histgram[hi] > Utilities::MaxNumber(series_num) * threshold_ratio) {
				series.push_back(hi);
				series_num.push_back(histgram[hi]);
			}
			else {
				if (series.size() < threshold_series_num) {
					for (int si = 0; si < series.size(); si++) {
						histgram[series[si]] = 0;
					}
				}

				series.clear();
				series_num.clear();
			}
		}
	}

	static void __drawHistgram__(Mat &histgram, vector<int> row_sum, vector<int> col_sum, pair<Point, Point> circumscribed_point)
	{
		for (int x = circumscribed_point.first.x; x < circumscribed_point.second.x; x++) {
			if (row_sum[x] != 0) {
				line(histgram, Point(x, 0), Point(x, row_sum[x] - 1), Scalar(0, 0, 255), 1, 4);
			}
		}

		for (int y = circumscribed_point.first.y; y < circumscribed_point.second.y; y++) {
			if (col_sum[y] != 0) {
				line(histgram, Point(0, y), Point(col_sum[y] - 1, y), Scalar(0, 0, 255), 1, 4);
			}
		}
	}

	static void __ComputeHistgram__(const Mat binarized_image, int min_x, int max_x, int min_y, int max_y, vector<int> &x_histgram, vector<int> &y_histgram)
	{
		for (int y = min_y; y < max_y; y++) {
			for (int x = min_x; x < max_x; x++) {
				if (binarized_image.at<uchar>(y, x) == 0) {
					x_histgram[x]++;
					y_histgram[y]++;
				}
			}
		}
	}

	static vector<pair<int, int>> __ComputeInterval__(vector<int> histgram)
	{
		bool is_series = false;
		int interval_begin = 0,
			interval_end = 0;
		vector<pair<int, int>> interval;

		for (int i = 0; i < histgram.size(); i++) {
			if (histgram[i] == 0) {
				if (is_series) {
					interval_end = i - 1;
					interval.push_back(
						make_pair(interval_begin, interval_end)
						);
				}

				is_series = false;
			}
			else {
				if (!is_series) {
					interval_begin = i;
				}

				is_series = true;
			}
		}

		return interval;
	}

	static void __drawInterval__(Mat &src, vector<pair<int, int>> interval_x, vector<pair<int, int>> interval_y)
	{
		for (int i = 0; i < interval_x.size(); i++) {
			line(src, Point(interval_x[i].first, 0), Point(interval_x[i].first, IMAGE_HEIGHT - 1), Scalar(0, 0, 255), 1, 4);
			line(src, Point(interval_x[i].second, 0), Point(interval_x[i].second, IMAGE_HEIGHT - 1), Scalar(0, 0, 255), 1, 4);
		}

		for (int i = 0; i < interval_y.size(); i++) {
			line(src, Point(0, interval_y[i].first), Point(IMAGE_WIDTH - 1, interval_y[i].first), Scalar(0, 0, 255), 1, 4);
			line(src, Point(0, interval_y[i].second), Point(IMAGE_WIDTH - 1, interval_y[i].second), Scalar(0, 0, 255), 1, 4);
		}
	}

	static void __drawAreas__(Mat &src, vector<pair<int, int>> interval_x, vector<pair<int, int>> interval_y)
	{
		for (int i = 0; i < interval_x.size(); i++) {
			rectangle(src, Point(interval_x[i].first, 0), Point(interval_x[i].second, IMAGE_HEIGHT - 1), Scalar(0, 0, 255), CV_FILLED, 4);
		}

		for (int i = 0; i < interval_y.size(); i++) {
			rectangle(src, Point(0, interval_y[i].first), Point(IMAGE_WIDTH - 1, interval_y[i].second), Scalar(0, 0, 255), CV_FILLED, 4);
		}
	}

	static void __drawOverlappedAreas__(Mat &src, vector<pair<int, int>> interval_x, vector<pair<int, int>> interval_y)
	{
		for (int i = 0; i < interval_x.size(); i++) {
			for (int j = 0; j < interval_y.size(); j++) {
				rectangle(src, Point(interval_x[i].first, interval_y[j].first), Point(interval_x[i].second, interval_y[j].second), Scalar(0, 0, 255), CV_FILLED, 4);
			}
		}
	}

};

int main(int argc, const char* argv[])
{
	vector<int> images = { 10 };

	for (int i = 0; i < images.size(); i++) {

		int image_no = images[i];

		//string image_path = "C://Users/sst/Pictures/res/p" + to_string(image_no) + ".jpg";
		string image_path = "C://Users/NEC-PCuser/Pictures/res/p" + to_string(image_no) + ".jpg";

		Mat orig_image,
			orig_image_grayscale,
			resized_image,
			resized_image_grayscale,
			process_image;

		/* ���摜�̓J���[�ƃO���[�X�P�[����ێ����Ă��� */
		orig_image = cv::imread(image_path, IMREAD_COLOR);
		orig_image_grayscale = cv::imread(image_path, IMREAD_GRAYSCALE);

		/* ���摜��640*480�Ƀ��T�C�Y */
		resize(orig_image, resized_image, Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, INTER_LINEAR);
		resize(orig_image_grayscale, resized_image_grayscale, Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, INTER_LINEAR);

		/* �����p�̉摜 */
		process_image = resized_image_grayscale.clone();

		/* ���摜��\�� */
		imshow("���摜", resized_image);
		waitKey();

		/*  */
		Mat blured_resized_image_grayscale;
		GaussianBlur(resized_image_grayscale, blured_resized_image_grayscale, Size(7, 7), 0);

		/* ���x�摜 */
		Mat value_image = GuideBoard::GetValue(resized_image);

		Mat blured_resized_image;
		GaussianBlur(resized_image, blured_resized_image, Size(7, 7), 0);

		Mat gray_image = blured_resized_image_grayscale.clone();
		Mat hue_image = GuideBoard::GetHue(blured_resized_image);
		Mat saturation_image = GuideBoard::GetSaturation(blured_resized_image);

		imshow("�ʓx", saturation_image);
		waitKey();

		Mat cannied_gray,
			cannied_hue;

		Canny(gray_image, cannied_gray, 50, 75);
		Canny(hue_image, cannied_hue, 50, 75);

		imshow("�F���摜�̃G�b�W�i�ʓx�̍l���O�j", cannied_hue);
		waitKey();

		/* �摜���ōʓx���Ⴂ�����ɃG�b�W������΂�������� */
		Mat sat_filtered_hue;
		vector<Point> delete_point;

		GuideBoard::RemoveEdgeOfLowSat(cannied_hue, saturation_image, sat_filtered_hue, 10, delete_point);

		imshow("�F���摜�̃G�b�W�i�ʓx�̍l����j", sat_filtered_hue);
		waitKey(); //imwrite("satAfter_" + to_string(i) + ".jpg", ~hueEdgeOfSat);


		/* �O���[�X�P�[���̃G�b�W�摜�ƐF���G�b�W�摜�̃G�b�W�������L����idilate�j */
		Mat delated_cannied_gray,
			delated_sat_filtered_hue;

		dilate(cannied_gray, cannied_gray, Mat());
		dilate(sat_filtered_hue, delated_sat_filtered_hue, Mat());

		/* 4�_�ŕ\�����̈�̍��W */
		vector< vector< Point > > apexes = GuideBoard::FindApexesByEdge(delated_sat_filtered_hue.clone(), 1000);

		if (apexes.size() == 0) {
			cout << "�l�p�`�����o�ł��܂���ł���\n";
			return 0;
		}

		vector<vector<int>> valueHists;

		imshow("����", value_image); waitKey();

		vector<Mat> sauvola_binarized;

		int valueMax = 0;
		int valueThr = 30;

		/* ���x�̃q�X�g�O�������Ƃ�Ȃ���A�ł������������x�l�̃C���f�b�N�X���擾 */
		for (int apexes_i = 0; apexes_i < apexes.size(); apexes_i++) {
			Mat binarized_image;

			//sauvolaFast(value, value, apexes[apexes_i], 0.15, 32);
			Utilities::sauvolaFast(value_image, binarized_image, apexes[apexes_i], 5, 0.25, 48);
			sauvola_binarized.push_back(binarized_image);

			/* ��������i�Ǝv����j���̗̂̈�i2�_�ŕ\������`�̈�j���擾 */
			//vector< pair<Point, Point> > object_area = GuideBoard::GetObjectArea(binarized_image, apexes[apexes_i], 5);
			vector< pair<Point, Point> > object_area = GuideBoard::GetObjectArea2(binarized_image, apexes[apexes_i], 5);

			/*vector<int> valueHist(256, 0);

			for (y = sRect[apexes_i][1].first; y <= sRect[apexes_i][1].second; y++) {
			for (x = sRect[apexes_i][0].first; x <= sRect[apexes_i][0].second; x++) {
			if (cn(apexes[apexes_i], Point(x, y))) {
			valueHist[value.at<uchar>(y, x)]++;
			}
			}
			}

			valueHists.push_back(valueHist);
			string t;

			for (int z = 0; z < valueHist.size(); z++) {
			t += to_string(z) + ',' + to_string(valueHist[z]) + '\n';
			}

			vector<int>::iterator iter = max_element(valueHist.begin(), valueHist.end());
			//valueMax.push_back(distance(valueHists[k].begin(), iter));
			valueMax = (int)(distance(valueHist.begin(), iter));

			for (y = sRect[apexes_i][1].first; y <= sRect[apexes_i][1].second; y++) {
			for (x = sRect[apexes_i][0].first; x <= sRect[apexes_i][0].second; x++) {
			if (cn(apexes[apexes_i], Point(x, y))) {
			if ((value.at<uchar>(y, x) > valueMax - valueThr) && (value.at<uchar>(y, x) < valueMax + valueThr)) {
			stock.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
			}
			else {
			stock.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
			}
			}
			else {
			stock.at<Vec3b>(y, x) = Vec3b(128, 128, 128);
			}
			}
			}*/
		}

		imwrite("a.png", sauvola_binarized[0]);

		imshow("����2", sauvola_binarized[0]); waitKey();


	}

	return 0;
}