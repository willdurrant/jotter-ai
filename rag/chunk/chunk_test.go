package chunk

import "testing"

func TestNormalizeWhitespace(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Python  was  created  in  February  20,1991", "Python was created in February 20,1991"},
		{"easy  to  \nread,\n \nwrite", "easy to read, write"},
		{"  leading and trailing  ", "leading and trailing"},
		{"already normal", "already normal"},
	}
	for _, c := range cases {
		if got := NormalizeWhitespace(c.in); got != c.want {
			t.Errorf("NormalizeWhitespace(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
