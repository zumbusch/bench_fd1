#!/usr/bin/perl

# Copyright (c) 2011, 2012, 2014, Gerhard Zumbusch
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * The names of its contributors may not be used to endorse or promote
#   products derived from this software without specific prior written
#   permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use strict;
use warnings;
use POSIX;



# run and extract flop rate
sub run {
    my $a = shift;
    my $f = 0;
    print $a." ";
    my $n = `$a`;
    my @w = split /\s+/, $n;
    foreach (@w) {
	if (/(flop=)/) {
	    my @p = split m!(=)!, $_;
	    $f = $p[2];
	}
    }
    return $f;
}

# run and extract error max
sub run_check {
    my $a = shift;
    my $f = "";
    print $a;
    my $n = `$a`;
    my @w = split /\s+/, $n;
    foreach (@w) {
	if (/(max=)/) {
	    $f = $_;
	}
    }
    print " error ".$f."\n";
    return $a." error ".$f."\n";
}

sub max {
    my $n0 = shift;
    my $f0 = shift;
    my $w0 = shift;
    my $t0 = shift;
    my $l0 = shift;
    my $n1 = shift;
    my $f1 = shift;
    my $w1 = shift;
    my $t1 = shift;
    my $l1 = shift;
    my @f;
    if ($f1 > $f0) {
	@f = ($n1, $f1, $w1, $t1, $l1);
	print "> ";
    } else {
	@f = ($n0, $f0, $w0, $t0, $l0);
    }
    return @f;
}

# tune parameter
sub find {
    my $g = shift; # defines
    my $w0 = shift; # algorithm minimum width
    my $w1 = shift; # algorithm maximum width
    my $t0 = shift; # algorithm minimum time steps
    my $t1 = shift; # algorithm maximum time steps
    my $l0 = shift; # algorithm minimum thread number
    my $l1 = shift; # algorithm maximum thread number
    my @fm = ("", 0, 0, 0, 0);
    # check
    print "check\n";
    my $rc = run_check("make run DEF=\"".$g."/8 -DWIDTH=".$w0." -DTIMESTEP=".$t0." -DLOCAL=".$l0." -DCHECK\"");
    # find parameter
    # coarse search
    print "coarse search\n";
    my $w;
    for ($w=$w0; $w<=$w1; $w*=2) {
	my $l;
	for ($l=$l0; $l<=$l1; $l*=2) {
	    my $ts;
	    for ($ts=$t0; $ts<=$t1; $ts*=2) {
		my $a= $g."/2 -DWIDTH=".$w." -DTIMESTEP=".$ts." -DLOCAL=".$l;
		my $f = run("make run DEF=\"".$a."\"");
		@fm = max(@fm, $a, $f, $w, $ts, $l);
		print "flop ".$f."\n";
	    }
	}
    }
    # one parameter search
    $w0 = int($fm[2] / 2);
    $w1 = 2 * $fm[2];
    my $ts = $fm[3];
    my $l = $fm[4];
    print "search width\n";
    foreach ($w0..$w1) {
	my $w = $_;
	my $a = $g." -DWIDTH=".$w." -DTIMESTEP=".$ts." -DLOCAL=".$l;
	my $f = run("make run DEF=\"".$a."\"");
	@fm = max(@fm, $a, $f, $w, $ts, $l);
	print "flop ".$f."\n";
    }
    # one parameter search
    $w = $fm[2];
    $t0 = 2 * int($fm[3] / 4); # even
    $t1 = 2 * $fm[3];
    $l = $fm[4];
    print "search timestep\n";
    for ($ts=$t0; $ts<=$t1; $ts+=2) {
 	my $a= $g." -DWIDTH=".$w." -DTIMESTEP=".$ts." -DLOCAL=".$l;
	my $f = run("make run DEF=\"".$a."\"");
	@fm = max(@fm, $a, $f, $w, $ts, $l);
	print "flop ".$f."\n";
    }
    # one parameter search
    $w = $fm[2];
    $ts = $fm[3];
    $l0 = int($fm[4] / 2);
    $l1 = 2 * $fm[4];
    print "search thread number\n";
    for ($l=$l0; $l<=$l1; $l+=8) {
 	my $a= $g." -DWIDTH=".$w." -DTIMESTEP=".$ts." -DLOCAL=".$l;
	my $f = run("make run DEF=\"".$a."\"");
	@fm = max(@fm, $a, $f, $w, $ts, $l);
	print "flop ".$f."\n";
    }

    # approx. narrow search interval
    $w0 = int($fm[2] * .9);
    $w1 = int($fm[2] * 1.1);
    $t0 = 2 * int($fm[3] * .45); # even
    $t1 = 2 * int($fm[3] * .55); # even
    $l0 = $fm[4];
    $l1 = $fm[4];
    print "fine search\n";

    foreach ($w0..$w1) {
	my $w = $_;
	my $l;
	for ($l=$l0; $l<=$l1; $l+=8) {
	    my $ts;
	    for ($ts=$t0; $ts<=$t1; $ts+=2) {
		my $a= $g." -DWIDTH=".$w." -DTIMESTEP=".$ts." -DLOCAL=".$l;
		my $f = run("make run DEF=\"".$a."\"");
		@fm = max(@fm, $a, $f, $w, $ts, $l);
		print "flop ".$f."\n";
	    }
	}
    }

    my $r = $fm[0]." flop ".$fm[1]."\n";
    print "maximum ",$r;
    return $rc.$r;
}

# run a set of optimizations
#change parameters accordingly
sub proc {
    my $f = shift; # enable double precision
    my $p = shift; # algorithm grid size
    my $d = shift; # def
    my @r;
    my $n = `make clean`;
    # single precision
    print "single precision\n";
    $r[0] = find ($d." -DFLOAT "."-DGRIDSIZE=".$p, 8, 100, 10, 100, 64, 256);
    if ($f == 2) {
	# double precision
	print "double precision\n";
	$r[1] = find ($d." -DGRIDSIZE=".int($p/2), 8, 64, 10, 100, 64, 256);
    }
    print "\nsummary\n";
    print @r;
    $n = `make clean`;
}

#----------------------------------------------------------------------
# memory/ grid size
#----------------------------------------------------------------------


# device mem
#proc (2, 1024*1024*256, "-DPROC=8 -DPFCPU");     # OpenCL CPU
#proc (2, 1024*1024*256, "-DPROC=16 -DDEV_MAX=2"); # 2 GPUs
#proc (2, 1024*1024*256, "-DPROC=5 -DREAL2 -DWRP=32");
#proc (2, 1024*1024*256, "-DPROC=5 -DWRP=32");
#proc (2, 1024*1024*256, "-DPROC=5 -DREAL2");
proc (2, 1024*1024*256, "-DPROC=5");





