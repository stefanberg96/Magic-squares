
int n, r;
long l,k;
int *index;
void init(int n1, int r1, long l1) {
	n = n1;
	r = r1;
	l = l1;
	k = 0;
	index = new int[r1];
	for (int i = 0; i < r1; i++) {
		index[i] = i;
	}
}

bool hasNext() {
	return k < l;
}

/**
* Finds the index which can be bumped up.
*/
int rightmostIndexBelowMax() {
	for (int i = r - 1; i >= 0; i--)
		if (index[i] < n - r + i) return i;
	return -1;
}

/**
* Move the index forward a notch.
*
* The algorithm finds the rightmost index element that can be incremented,
* increments it, and then changes the elements to the right to each be 1
* plus the element on their left.
*
* For example, if an index of 5C3 at a time is at [0, 3, 4], only the 0 can
* be incremented without running out of room. The next index is [1, 1+1, 1+2]
* or [1, 2, 3]. This will be followed by [1, 2, 4], [1, 3, 4], and [2, 3, 4].
*/
void moveIndex() {
	int i = rightmostIndexBelowMax();
	if (i >= 0) {
		index[i] = index[i] + 1;
		for (int j = i + 1; j < r; j++)
			index[j] = index[j - 1] + 1;
	}
}

int* next() {
	if (!hasNext()) {
		index[0] = -1;
		return index;
	}
	if (k++ > 0) moveIndex();
	return index;
}

long choose(int x, int y) {
	long *result = new long;
	if (x < y || x<0) {
		*result = 0;
		return *result;
	}
	if (x == y) {
		*result = 1;
		return *result;
	}
	int s = x < (x - y) ? x : (x - y);
	long t = x;
	long *a = new long;
	long *b = new long;
	for (*a = x - 1, *b = 2; *b <= s; --*a, ++*b) {
		t = (t * (*a)) / (*b);
	}
	*result = t;
	return *result;
}

