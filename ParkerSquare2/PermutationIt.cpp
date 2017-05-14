
int *arr = new int[9];
int *p = new int[9];
int i = 1;
bool didFindNext;

int * initPerm() {
	for (int j = 0; j < 9; j++) {
		arr[j] = j;
		p[j] = 0;
	}
	return arr;
}

void swap(int x, int y) {
	int temp = arr[x];
	arr[x] = arr[y];
	arr[y] = temp;
}

bool findNext() {
	while (i < 9) {
		if (p[i] < i) {
			int j = (i % 2 == 1) ? p[i] : 0;
			swap(j, i);
			p[i]++;
			i = 1;
			return didFindNext = true;
		}
		else {
			p[i] = 0;
			i++;
		}
	}
	return false;
}

bool hasNextPerm() {
	return didFindNext || findNext();
}

void nextPerm() {
	if (!didFindNext) {
		findNext();
	}

	didFindNext = false;
	return;
}