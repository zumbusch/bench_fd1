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

sub max {
    my $n0 = shift;
    my $f0 = shift;
    my $w0 = shift;
    my $t0 = shift;
    my $n1 = shift;
    my $f1 = shift;
    my $w1 = shift;
    my $t1 = shift;
    my @f;
    if ($f1 > $f0) {
	@f = ($n1, $f1, $w1, $t1);
	print "> ";
    } else {
	@f = ($n0, $f0, $w0, $t0);
    }
    return @f;
}

# tune parameter
sub find {
    my $c = shift; # compiler + flags
    my $a = shift; # algorithm grid size
    my $w0 = shift; # algorithm minimum width
    my $w1 = shift; # algorithm maximum width
    my $t0 = shift; # algorithm minimum time steps
    my $t1 = shift; # algorithm maximum time steps
    my @fm = ("", 0, 0, 0);
    # check
    print "check\n";
    my $rc; # = run_check("make run ".$c." DEF=\"-DWIDTH=".$w0." -DGRIDSIZE=".(1024*10)." -DTIMESTEP=4 -DCHECK\"");
    # find parameter
    # coarse search
    print "coarse search\n";
    foreach ($w0..$w1) {
	my $w = $_;
	my $unr = "-DWIDTH=".$_;
	my $ts;
	my $inc = 2;
	for ($ts=$t0; $ts<=$t1; $ts+=$inc) {
	    if ($ts >= 4*$inc) {
		$inc *= 2;
	    }
	    my $a= $unr." -DTIMESTEP=".$ts." ".$a;
	    my $f = run("make run ".$c." DEF=\"".$a."\"");
	    @fm = max(@fm, $a, $f, $w, $ts);
	    print "flop ".$f."\n";
	}
    }
    $w0 = $fm[2];
    $w1 = $fm[2];
    # approx. narrow search interval
    $t0 = 2 * int($fm[3] * 0.3) - 20;
    if ($t0 <2) {
	$t0 = 2;
    }
    $t1 = 2 * int($fm[3] * 0.8) + 20;
    print "fine search\n";
    foreach ($w0..$w1) {
	my $w = $_;
	my $unr = "-DWIDTH=".$_;
	my $ts;
	my $inc = 2;
	for ($ts=$t0; $ts<=$t1; $ts+=$inc) {
	    my $a= $unr." -DTIMESTEP=".$ts." ".$a;
	    my $f = run("make run ".$c." DEF=\"".$a."\"");
	    @fm = max(@fm, $a, $f, $w, $ts);
	    print "flop ".$f."\n";
	}
    }
    my $r = $c." ".$fm[0]." flop ".$fm[1]."\n";
    print "maximum ",$r;
    return $rc.$r;
}

# run a set of optimizations
sub proc {
    my $c = shift; # compiler + flags
    my $o = shift; # LIB, RUN options, run cross-compiled code
    my $w = shift; # algorithm WIDTH
    my $f = shift; # enable double precision
    my $p = shift; # algorithm grid size
    my $g = "-DGRIDSIZE=".$p;
    my @r;
    my $n = `make clean`;
    # single precision
    print "single precision\n";
    $r[0] = find ("CC=\"".$c." -DFLOAT\" ".$o, $g, 2, 24, 2, 4);
    #$r[0] = find ("CC=\"".$c." -DFLOAT\" ".$o, $g, $w, $w, 2, 32);
    print "\nsummary\n";
    print @r;
    $n = `make clean`;
}

#----------------------------------------------------------------------
# choose processor
# native or cross compiler
# memory/ grid size
#----------------------------------------------------------------------


# Cell BE SPU
proc ("spu-g++ -O3 -DSPU", "", 20, 1, 1024*20);


